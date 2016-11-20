FROM anaconda-tfgpu-devel

# Virtual env
RUN conda create --name keras-gym python=2.7
RUN source activate keras-gym \
	&& conda update -y conda 
RUN source activate keras-gym \
	&& conda update -y anaconda

# Tensorflow
RUN source activate keras-gym \
	&& pip install --upgrade /tensorflow/pip/tensorflow-*.whl

# Keras
RUN source activate keras-gym \
	&& conda install -y quandl pyyaml
RUN source activate keras-gym \
	&& pip install --no-deps git+git://github.com/fchollet/keras.git

# Open AI Gym
RUN source activate keras-gym \
	&& conda install -y requests
RUN source activate keras-gym \
	&& pip install pyglet gym

