import ast
import logging
import os
import re
import tarfile
from urllib.error import URLError
from urllib.request import urlretrieve

import markdown
import requests
from appdirs import AppDirs
from bs4 import BeautifulSoup as Soup

APP_AUTHOR = "jasonrig"
APP_NAME = "transfer-learning-demo"
SLIM_README_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/slim/README.md"
SLIM_MODEL_BASE_URL = "https://github.com/tensorflow/models/blob/master/research/slim/nets/"
IMAGENET_MAPPINGS_URL = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/c2c91c8e767d04621020c30ed31192724b863041/imagenet1000_clsid_to_human.txt"

logger = logging.getLogger(__name__)


def get_dirs():
    """
    Creates a series of application directories and returns their paths. The user_cache_dir path is guaranteed to exist
    and be writable.
    :return: a dictionary of application paths
    """
    dirs = AppDirs(APP_NAME, APP_AUTHOR)
    dir_types = [dir_type for dir_type in dir(dirs) if dir_type.endswith("_cache_dir")]
    created_dirs = {}
    for dir_type in dir_types:
        new_dir = getattr(dirs, dir_type)
        try:
            if not os.path.isdir(new_dir):
                logger.debug("App directory (%s) %s does not exist. Trying to create it." % (dir_type, new_dir))
                os.makedirs(new_dir)
                logger.debug("Created app directory (%s) %s" % (dir_type, new_dir))
            else:
                logger.debug("App directory (%s) %s already exists, skipping." % (dir_type, new_dir))
            created_dirs[dir_type] = new_dir
        except OSError as e:
            logger.debug("Could not create app directory (%s) because an exception was thrown: %s" % (dir_type, e))

    assert 'user_cache_dir' in created_dirs.keys(), "`user_cache_dir` was not created but was expected to be here: %s" % dirs.user_cache_dir
    assert os.access(created_dirs['user_cache_dir'], os.W_OK), "Expected %s to be writable, but it isn't." % \
                                                               created_dirs['user_cache_dir']

    return created_dirs


def pretrained_model_urls():
    """
    Checks the TensorFlow models repo and scrapes URLs from the README.md file to pretrained parameters for all
    supported models
    :return: a dictionary of model names and their pretrained parameter URLs
    """

    def _get_model_name_and_url(row):
        links = row.find_all('a')
        cells = row.find_all('td')
        has_link_to_model_source = any(map(lambda l: l.get('href').startswith(SLIM_MODEL_BASE_URL), links))

        if not has_link_to_model_source or len(cells) == 0:
            return None

        link_to_tgz = [
            a.get('href') for a in filter(
                lambda l: l.get('href').lower().endswith(".tgz") or l.get('href').lower().endswith(".tar.gz"),
                links)
        ]
        if len(link_to_tgz) == 0:
            return None
        assert len(link_to_tgz) == 1, "Found more than one gzipped archive but expected only one"

        return re.sub(r'[^A-Za-z0-9._\-\s]', '', cells[0].text), link_to_tgz[0]

    readme_markdown = requests.get(SLIM_README_URL).text
    readme_html = markdown.markdown(readme_markdown, extensions=['markdown.extensions.tables'])
    soup = Soup(readme_html, 'html.parser')
    table_rows = soup.find_all('tr')

    assert len(table_rows) > 0, "Could not find any tables referencing TF-Slim pretrained models in README.md"

    models = filter(lambda x: x is not None, [_get_model_name_and_url(m) for m in table_rows])
    return dict(models)


def get_model_checkpoint(model_name, target_file=None, download_dir_type='user_cache_dir'):
    """
    Returns a path to the TensorFlow checkpoint file containing pretrained parameters for the given model. This might
    involve downloading the tar file or it could be available in the cache.
    :param model_name: name of a supported model
    :param target_file: optional name of the expected target checkpoint file
    :param download_dir_type: optional target directory type
    :return: a path to the checkpoint file
    """
    dirs = get_dirs()
    assert download_dir_type in dirs.keys(), "`download_dir_type` was '%s' but must be one of %s" % (
        download_dir_type, ', '.join(dirs.keys()))

    try:
        model_url = pretrained_model_urls()[model_name]
    except KeyError:
        logger.error("No pretrained model found for %s" % model_name)
        raise

    logger.debug("Trying to find pretrained parameters for %s" % model_name)

    source_file = model_url.split("/")[-1]
    if target_file is None:
        target_file = "%s.ckpt" % re.sub(r'(\.tar\.gz)|(\.tgz)$', '', source_file)

    logger.debug("Looking for %s on disk..." % target_file)

    for dir_type, path in dirs.items():
        test_path = os.path.join(path, target_file)
        if os.path.isfile(test_path):
            logger.debug("Found pretrained parameters %s" % test_path)
            return test_path
        else:
            logger.debug("Pretrained parameters were not found in %s" % path)

    logger.warning("No pretrained parameters were found on disk. Attempting to download from %s" % model_url)
    destination_dir = dirs[download_dir_type]
    ckpt_destination = os.path.join(destination_dir, target_file)
    tar_destination = maybe_download(model_url, destination_dir, source_file)

    assert os.path.isfile(tar_destination), "Expected %s to exist but could not be found" % tar_destination

    logger.debug("Archive successfully downloaded. Decompressing...")

    with tarfile.open(tar_destination) as tar:
        ckpt_file_in_tar = [item for item in tar.getnames() if item.endswith(".ckpt")]

        assert len(ckpt_file_in_tar) == 1, "Expected to find only one checkpoint file in %s but found %i" % (
        tar_destination, len(ckpt_file_in_tar))

        ckpt_file_in_tar = ckpt_file_in_tar[0]
        try:
            tar.extract(ckpt_file_in_tar, destination_dir)
            extract_location = os.path.join(destination_dir, ckpt_file_in_tar)
            assert os.path.isfile(
                extract_location), "Expected checkpoint to be extracted as %s, but it wasn't." % extract_location
            os.rename(extract_location, ckpt_destination)
            logger.debug("Checkpoint file %s extracted to %s successfully" % (ckpt_file_in_tar, ckpt_destination))
        except tarfile.TarError as e:
            logger.error("An error was encountered while extracting the checkpoint file: %s" % e)
            raise

    logger.debug("Cleaning up downloaded tar file.")
    delete_file_safely(tar_destination)

    return ckpt_destination


def get_imagenet_mappings():
    """
    Fetches a dictionary of class id to description mappings from github
    :return: imagenet class mapping dictionary
    """
    with open(maybe_download(IMAGENET_MAPPINGS_URL), 'r') as f:
        return ast.literal_eval(f.read())


def maybe_download(url, destination_dir=None, target_file_name=None, invalidate_cache=False):
    """
    Downloads a file from the given URL or returns a cached version
    :param url: URL from which to download the file
    :param destination_dir: directory to which the file should be downloaded, defaults to user_cache_dir
    :param target_file_name: the name given to the downloaded file
    :return: path to the downloaded file
    """
    destination_dir = destination_dir or get_dirs()['user_cache_dir']
    target_file_name = target_file_name or url.split("/")[-1]
    file_path = os.path.join(destination_dir, target_file_name)

    if invalidate_cache:
        delete_file_safely(file_path)

    if not os.path.isfile(file_path):
        try:
            urlretrieve(url, file_path)
        except URLError as e:
            logger.error("Could not download file from %s: %s" % (url, e))
            delete_file_safely(file_path)
            raise
    else:
        logger.debug("File already exists - no need to download.")
    return file_path


def delete_file_safely(file_path, raise_exception_if_not_found=False):
    """
    Deletes a file only if it resides in any of the app directories
    :param file_path: path to file to delete
    :param raise_exception_if_not_found: if False, silently fails if file doesn't exist
    """
    in_path = any([os.path.abspath(file_path).startswith(d) for d in get_dirs().values()])
    assert in_path, "Expected file %s to be in one of %s but it wasn't." % (file_path, ', '.join(get_dirs().values()))
    assert not os.path.isdir(file_path), "Expected `file_path` to be a file, but found a directory instead."
    try:
        os.remove(file_path)
    except FileNotFoundError:
        if raise_exception_if_not_found:
            raise
