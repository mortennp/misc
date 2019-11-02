#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

FROM rocker/tidyverse:latest

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Non-root user
ARG USERNAME=rstudio
ARG USER_UID=1001
ARG USER_GID=$USER_UID
# The rocker/tidyverse image comes with a base non-root 'rstudio' user.
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME 
#
# For Linux, non-root user's GID/UID must match your local
# user UID/GID to avoid permission issues with bind mounts. Update USER_UID / USER_GID 
# if yours is not 1000. See https://aka.ms/vscode-remote/containers/non-root-user.
RUN if [ "$USER_GID" != "1000" ]; then groupmod $USERNAME --gid $USER_GID; fi \
    && if [ "$USER_UID" != "1000" ]; then usermod --uid $USER_UID $USERNAME; fi 
#
# Make vscode dirs and set permissions
RUN mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown ${USER_UID}:${USER_GID} /home/$USERNAME/.vscode-server* 
#
# Add sudo support. Omit if you don't need to install software after connecting.
RUN apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Configure apt and install packages need by vscode R extension
RUN apt-get -y install --no-install-recommends gnupg apt-transport-https \
    && wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > microsoft.asc.gpg \
    && mv microsoft.asc.gpg /etc/apt/trusted.gpg.d/ \
    && wget -q https://packages.microsoft.com/config/debian/9/prod.list \
    && mv prod.list /etc/apt/sources.list.d/microsoft-prod.list \
    && chown root:root /etc/apt/trusted.gpg.d/microsoft.asc.gpg \
    && chown root:root /etc/apt/sources.list.d/microsoft-prod.list
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-transport-https 2>&1 
RUN apt-get update \
    && apt-get -y install --no-install-recommends dotnet-runtime-3.0 2>&1     
# RUN apt-get update \
#     && apt-get -y install --no-install-recommends apt-utils apt-transport-https software-properties-common dialog 2>&1 \
#     #
#     # install git iproute2, process tools, lsb-release (common in install instructions for CLIs) and libzip for R Tools extension
#     && apt-get -y install git iproute2 procps lsb-release libzip-dev \
#     #
#     # Register Microsoft key and feed
#     && wget -q https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb \
#     && dpkg -i packages-microsoft-prod.deb \
#     #
#     # Install the .NET Runtime
#     && add-apt-repository universe \
#     && apt-get install apt-transport-https \
#     && apt-get update \
#     && apt-get -y install dotnet-runtime-2.1 \
#     #
#     # Clean up
#     && apt-get autoremove -y \
#     && apt-get clean -y \
#     && rm -rf /var/lib/apt/lists/*

# Set the default user. Omit if you want to keep the default as root.
USER $USERNAME
# Done with -u in devcontainer.json

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=

CMD ["/bin/bash"]