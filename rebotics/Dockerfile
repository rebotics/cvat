ARG PIP_VERSION=22.0.2
ARG BASE_IMAGE=ubuntu:22.04

FROM ${BASE_IMAGE} as build-image-base

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq \
        curl \
        g++ \
        gcc \
        git \
        libgeos-dev \
        libldap2-dev \
        libsasl2-dev \
        make \
        nasm \
        pkg-config \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_VERSION
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN --mount=type=cache,target=/root/.cache/pip/http \
    python3 -m pip install -U pip==${PIP_VERSION}

# We build OpenH264, FFmpeg and PyAV in a separate build stage,
# because this way Docker can do it in parallel to all the other packages.
FROM build-image-base AS build-image-av

# Compile Openh264 and FFmpeg
ARG PREFIX=/opt/ffmpeg
ARG PKG_CONFIG_PATH=${PREFIX}/lib/pkgconfig

ENV FFMPEG_VERSION=4.3.1 \
    OPENH264_VERSION=2.1.1

WORKDIR /tmp/openh264
RUN curl -sL https://github.com/cisco/openh264/archive/v${OPENH264_VERSION}.tar.gz --output - | \
    tar -zx --strip-components=1 && \
    make -j5 && make install-shared PREFIX=${PREFIX} && make clean

WORKDIR /tmp/ffmpeg
RUN curl -sL https://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.gz --output - | \
    tar -zx --strip-components=1 && \
    ./configure --disable-nonfree --disable-gpl --enable-libopenh264 \
        --enable-shared --disable-static --disable-doc --disable-programs --prefix="${PREFIX}" && \
    make -j5 && make install && make clean

COPY utils/dataset_manifest/requirements.txt /tmp/utils/dataset_manifest/requirements.txt

# Since we're using pip-compile-multi, each dependency can only be listed in
# one requirements file. In the case of PyAV, that should be
# `dataset_manifest/requirements.txt`. Make sure it's actually there,
# and then remove everything else.
RUN grep -q '^av==' /tmp/utils/dataset_manifest/requirements.txt
RUN sed -i '/^av==/!d' /tmp/utils/dataset_manifest/requirements.txt

# Work around https://github.com/PyAV-Org/PyAV/issues/1140
RUN pip install setuptools wheel 'cython<3'

RUN --mount=type=cache,target=/root/.cache/pip/http \
    python3 -m pip wheel --no-binary=av --no-build-isolation \
    -r /tmp/utils/dataset_manifest/requirements.txt \
    -w /tmp/wheelhouse

# This stage builds wheels for all dependencies (except PyAV)
FROM build-image-base AS build-image

COPY cvat/requirements/ /tmp/cvat/requirements/
COPY utils/dataset_manifest/requirements.txt /tmp/utils/dataset_manifest/requirements.txt
COPY rebotics/requirements.txt /tmp/rebotics/requirements.txt

# Exclude av from the requirements file
RUN sed -i '/^av==/d' /tmp/utils/dataset_manifest/requirements.txt

ARG CVAT_CONFIGURATION="production"

RUN --mount=type=cache,target=/root/.cache/pip/http \
    DATUMARO_HEADLESS=1 python3 -m pip wheel --no-deps \
    -r /tmp/cvat/requirements/${CVAT_CONFIGURATION}.txt \
    -r /tmp/rebotics/requirements.txt \
    -w /tmp/wheelhouse

FROM golang:1.20.5 AS build-smokescreen

RUN git clone --depth=1 -b v0.0.4 https://github.com/stripe/smokescreen.git
RUN cd smokescreen && go build -o /tmp/smokescreen

FROM node:lts-slim AS cvat-ui

ARG WA_PAGE_VIEW_HIT
ARG UI_APP_CONFIG
ARG CLIENT_PLUGINS
ARG DISABLE_SOURCE_MAPS
ARG SOURCE_MAPS_TOKEN

ENV TERM=xterm \
    LANG='C.UTF-8'  \
    LC_ALL='C.UTF-8'

# Install dependencies
COPY package.json /tmp/
COPY yarn.lock /tmp/
COPY cvat-core/package.json /tmp/cvat-core/
COPY cvat-canvas/package.json /tmp/cvat-canvas/
COPY cvat-canvas3d/package.json /tmp/cvat-canvas3d/
COPY cvat-ui/package.json /tmp/cvat-ui/
COPY cvat-data/package.json /tmp/cvat-data/

# Install common dependencies
WORKDIR /tmp/
RUN DISABLE_HUSKY=1 yarn --frozen-lockfile

# Build source code
COPY cvat-data/ /tmp/cvat-data/
COPY cvat-core/ /tmp/cvat-core/
COPY cvat-canvas3d/ /tmp/cvat-canvas3d/
COPY cvat-canvas/ /tmp/cvat-canvas/
COPY cvat-ui/ /tmp/cvat-ui/
RUN CLIENT_PLUGINS="${CLIENT_PLUGINS}" \
DISABLE_SOURCE_MAPS="${DISABLE_SOURCE_MAPS}" \
UI_APP_CONFIG="${UI_APP_CONFIG}" \
SOURCE_MAPS_TOKEN="${SOURCE_MAPS_TOKEN}" yarn run build:cvat-ui

FROM ${BASE_IMAGE}

ARG http_proxy
ARG https_proxy
ARG no_proxy="nuclio,${no_proxy}"
ARG socks_proxy
ARG TZ="Etc/UTC"

ENV TERM=xterm \
    http_proxy=${http_proxy}   \
    https_proxy=${https_proxy} \
    no_proxy=${no_proxy} \
    socks_proxy=${socks_proxy} \
    LANG='C.UTF-8'  \
    LC_ALL='C.UTF-8' \
    TZ=${TZ}

ARG USER="django"
ARG CVAT_CONFIGURATION="production"
ENV DJANGO_SETTINGS_MODULE="cvat.settings.${CVAT_CONFIGURATION}"

# Install necessary apt packages
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get --no-install-recommends install -yq \
        bzip2 \
        ca-certificates \
        curl \
        git \
        libgeos-c1v5 \
        libgl1 \
        libgomp1 \
        libldap-2.5-0 \
        libpython3.10 \
        libsasl2-2 \
        nginx \
        p7zip-full \
        poppler-utils \
        python3 \
        python3-venv \
        supervisor \
        tzdata \
    && ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/* && \
    echo 'application/wasm wasm' >> /etc/mime.types

# Install smokescreen
COPY --from=build-smokescreen /tmp/smokescreen /usr/local/bin/smokescreen

# Add a non-root user
ENV USER=${USER}
ENV HOME /home/${USER}
RUN adduser --shell /bin/bash --disabled-password --gecos "" ${USER}

ARG CLAM_AV="no"
RUN if [ "$CLAM_AV" = "yes" ]; then \
        apt-get update && \
        apt-get --no-install-recommends install -yq \
            clamav \
            libclamunrar9 && \
        sed -i 's/ReceiveTimeout 30/ReceiveTimeout 300/g' /etc/clamav/freshclam.conf && \
        freshclam && \
        chown -R ${USER}:${USER} /var/lib/clamav && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install wheels from the build image
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ARG PIP_VERSION
ARG PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m pip install -U pip==${PIP_VERSION}
RUN --mount=type=bind,from=build-image,source=/tmp/wheelhouse,target=/mnt/wheelhouse \
    --mount=type=bind,from=build-image-av,source=/tmp/wheelhouse,target=/mnt/wheelhouse-av \
    python -m pip install --no-index /mnt/wheelhouse/*.whl /mnt/wheelhouse-av/*.whl

ENV NUMPROCS=1
COPY --from=build-image-av /opt/ffmpeg/lib /usr/lib

# These variables are required for supervisord substitutions in files
# This library allows remote python debugging with VS Code
ARG CVAT_DEBUG_ENABLED
RUN if [ "${CVAT_DEBUG_ENABLED}" = 'yes' ]; then \
        python3 -m pip install --no-cache-dir debugpy; \
    fi

# Install and initialize CVAT, copy all necessary files
COPY cvat/nginx.conf /etc/nginx/nginx.conf
COPY --chown=${USER} components /tmp/components
COPY --chown=${USER} supervisord/ ${HOME}/supervisord
COPY --chown=${USER} wait-for-it.sh manage.py backend_entrypoint.sh wait_for_deps.sh ${HOME}/
COPY --chown=${USER} utils/ ${HOME}/utils
COPY --chown=${USER} cvat/ ${HOME}/cvat
COPY --chown=${USER} rqscheduler.py ${HOME}

# Install UI
COPY --from=cvat-ui --chown=${USER} /tmp/cvat-ui/dist ${HOME}/static-ui
RUN sed -i -E "s#(<\!doctype html>)#{% load static %}\1#g" ${HOME}/static-ui/index.html && \
    perl -pe "s#\"/(assets/.*?)\"#\"{% static \"\1\" %}\"#g" -i ${HOME}/static-ui/index.html && \
    mkdir -p ${HOME}/cvat/apps/rebotics/templates && \
    mv ${HOME}/static-ui/index.html ${HOME}/cvat/apps/rebotics/templates

# Override supervisor / nginx configs and some scripts
COPY --chown=${USER} rebotics/supervisord.conf ${HOME}/supervisord/all.conf
COPY --chown=${USER} rebotics/backend_entrypoint.sh rebotics/wait_for_deps.sh ${HOME}/
COPY rebotics/nginx.conf /etc/nginx/nginx.conf

# Add version file
COPY --chown=${USER} VERSION ${HOME}/static-ui/version.txt

ARG COVERAGE_PROCESS_START
RUN if [ "${COVERAGE_PROCESS_START}" ]; then \
        echo "import coverage; coverage.process_startup()" > /opt/venv/lib/python3.10/site-packages/coverage_subprocess.pth; \
    fi

# RUN all commands below as 'django' user
USER ${USER}
WORKDIR ${HOME}

RUN mkdir -p data share keys logs /tmp/supervisord static

EXPOSE 8080
ENTRYPOINT ["./backend_entrypoint.sh"]
