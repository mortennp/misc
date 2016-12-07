FROM cpu-keras-gym

# Basics
RUN apt-get update && apt-get install -y \
	curl \
	wget \
	build-essential \
	git

# Geode
# 	python python-numpy
RUN apt-get install -y scons libgmp-dev
# 	python-scipy python-pytest  
RUN apt-get install -y libpng-dev libjpeg-dev libopenexr-dev
RUN apt-get install -y libboost-all-dev 
RUN apt-get install -y libatlas-base-dev

RUN git clone https://github.com/otherlab/geode.git
RUN source activate keras-gym \
	&& cd geode && scons -j 5
RUN source activate keras-gym \
	&& cd geode && scons install
RUN source activate keras-gym \
	&& cd geode && python setup.py install
RUN source activate keras-gym \
	&& cd geode && py.test

# Pentago
RUN apt-get install -y scons openmpi-bin libopenmpi-dev zlib1g-dev liblzma-dev libsnappy-dev

RUN git clone https://github.com/girving/pentago.git

RUN source activate keras-gym \
	&& cd pentago && scons -j 5
RUN source activate keras-gym \
	&& cd pentago && py.test
RUN source activate keras-gym \
	&& cd pentago && scons install
RUN source activate keras-gym \
	&& cd pentago && python setup.py install
