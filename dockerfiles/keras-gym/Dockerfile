FROM cuda-tf-anaconda

#RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Virtual env
RUN conda create --name keras-gym python=2.7
#RUN source activate keras-gym \
#	&& conda update -y conda 
#RUN source activate keras-gym \
#	&& conda update -y anaconda

# Tensorflow
RUN source activate keras-gym \
	&& conda install -y six funcsigs pbr mock setuptools protobuf numpy 
ENV TF_BINARY_URL https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.11.0-cp27-none-linux_x86_64.whl
RUN source activate keras-gym \
	&& pip install --ignore-installed --upgrade ${TF_BINARY_URL}

# Keras
RUN source activate keras-gym \
	&& conda install -y quandl pyyaml scipy h5py
RUN source activate keras-gym \
	&& pip install --no-deps git+git://github.com/fchollet/keras.git

# Open AI Gym
RUN source activate keras-gym \
	&& conda install -y requests
RUN source activate keras-gym \
	&& pip install pyglet gym

RUN sudo apt-get install zlib1g-dev cmake
RUN source activate keras-gym \
	&& pip install gym[atari]

# if "Intel MKL FATAL ERROR: Cannot load libmkl_avx.so or libmkl_def.so.":
#	conda install -f mkl mkl-service
#	conda install -f numpy scipy scikit-learn scikit-image numexpr
