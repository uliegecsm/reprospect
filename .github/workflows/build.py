import argparse
import json
import logging
import os
import typing

STAGING_INFIX: typing.Final[str] = '-staging-pr-'

def to_output(value: str) -> None:
    logging.info(value)
    print(value)

def bool_to_str(value: bool) -> str: # ruff:ignore[boolean-type-hint-positional-argument]
    return 'true' if value else 'false'

def str_to_bool(value: str) -> bool:
    match value:
        case 'true':
            return True
        case 'false':
            return False
        case _:
            raise ValueError(value)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup-job-matrix-image-changed-files', type=str_to_bool, required=True)
    parser.add_argument('--build-images-changed-files', type=str_to_bool, required=True)
    return parser.parse_args()

def main(*, args: argparse.Namespace) -> None:
    """
    Decide whether to push, and with which staging suffix (if any).
    """
    GITHUB_CONTEXT: typing.Final[dict] = json.loads(os.environ['GITHUB_CONTEXT'])

    # Are we on the default branch?
    DEFAULT_BRANCH: typing.Final[str] = GITHUB_CONTEXT['event']['repository']['default_branch']
    DEFAULT_REF: typing.Final[str] = f'refs/heads/{DEFAULT_BRANCH}'
    logging.info(f'The default branch is {DEFAULT_BRANCH!r} ({DEFAULT_REF!r}).')

    IS_DEFAULT_BRANCH: typing.Final[bool] = GITHUB_CONTEXT['ref'] == DEFAULT_REF

    if IS_DEFAULT_BRANCH:
        logging.info(f'The current \'ref\' is ${GITHUB_CONTEXT["ref"]!r}, which is the default branch.')
    else:
        logging.info(f'The current \'ref\' is {GITHUB_CONTEXT["ref"]!r}.')

    # By default, push is false.
    push: bool = False

    # By default, the suffix is empty.
    suffix: str = ''

    # By default, the number of retention days for Docker build record artifact is 1.
    # See also https://github.com/docker/build-push-action.
    docker_build_record_retention_days: int = 1

    # Images generally need a rebuild either because of file changes or it is requested with a workflow dispatch.
    # Otherwise, the jobs can be skipped, thus speeding up the workflow.
    FORCE_BUILD_IMAGES: typing.Final[bool] = GITHUB_CONTEXT['event_name'] == 'workflow_dispatch' and GITHUB_CONTEXT['event']['inputs']['build-images'] == 'true'
    logging.info(f'Force rebuild of images is set to {FORCE_BUILD_IMAGES!r} (event is {GITHUB_CONTEXT["event_name"]!r}).')

    setup_job_matrix_image: bool = args.setup_job_matrix_image_changed_files or FORCE_BUILD_IMAGES
    build_images: bool = args.build_images_changed_files or FORCE_BUILD_IMAGES

    logging.info(f'The \'setup-job-matrix-image\' condition is {setup_job_matrix_image!r}.')
    logging.info(f'The \'build-images\' condition is {build_images!r}.')

    # Canonical images (i.e. without staging infix) are only ever pushed from the default branch.
    match GITHUB_CONTEXT['event_name']:
        case 'workflow_dispatch':
            if IS_DEFAULT_BRANCH and (setup_job_matrix_image or build_images):
                push = True
                docker_build_record_retention_days = 0

        case 'schedule' | 'push' | 'release':
            if IS_DEFAULT_BRANCH and (setup_job_matrix_image or build_images):
                push = True
                docker_build_record_retention_days = 0

        case 'pull_request':
            push = True
            suffix = f'{STAGING_INFIX}{GITHUB_CONTEXT["event"]["pull_request"]["number"]}'

        case _:
            raise ValueError(GITHUB_CONTEXT['event_name'])

    to_output(f'image-push={push}')
    to_output(f'docker-build-record-retention-days={docker_build_record_retention_days}')

    to_output(f'setup-job-matrix-image={bool_to_str(setup_job_matrix_image)}')
    to_output(f'setup-job-matrix-image-staging-tag-suffix={suffix if setup_job_matrix_image else ""}')

    to_output(f'build-images={bool_to_str(build_images)}')
    to_output(f'build-images-staging-tag-suffix={suffix if build_images else ""}')

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    logging.info(f"Received arguments: {args}")

    main(args=args)
