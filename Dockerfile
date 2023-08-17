FROM buildpack-deps:jammy

# install stuff
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    cmake=3.22.1-1ubuntu1 \
    libmpich-dev=4.0-3 \
    mpich=4.0-3 \
    doxygen=1.9.1-2ubuntu2 \
    libboost-dev=1.74.0.3ubuntu7 \
    libgsl-dev=2.7.1+dfsg-3 \
    fonts-humor-sans=1.0-4 \
    cython3=0.29.28-1ubuntu3 \
    python3-dev=3.10.4-0ubuntu2 \
    python3-pip=22.0.2+dfsg-1 \
    python3-scipy=1.8.0-1exp2ubuntu1 \
    python3-pandas=1.3.5+dfsg-3 \
    python3-seaborn=0.11.2-3 \
    python3-astropy=5.0.2-1 \
    python3-yaml=5.4.1-1ubuntu1 \
    python3-sympy=1.9-1 \
    python3-matplotlib=3.5.1-2build1 \
    ipython3=7.31.1-1 \
    jupyter=4.9.1-1 \
    jupyter-notebook=6.4.8-1 \
    antlr4=4.7.2-5 \
    libhdf5-dev=1.10.7+repack-4ubuntu2 \
    bison=2:3.8.2+dfsg-1build1 \
    flex=2.6.4-8build2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# fix some executable names
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    update-alternatives --install /usr/bin/ipython ipython /usr/bin/ipython3 10

# install mpi4py and h5py (deb packages may depend on libopenmpi which we do not want)
RUN pip --no-cache-dir install --no-deps mpi4py==3.1.3 && \
    pip --no-cache-dir install --no-deps h5py==3.7.0

# Install NEST 3.3
RUN git clone --depth 1 -b v3.3 https://github.com/nest/nest-simulator /usr/src/nest-simulator && \
  mkdir nest-build && \
  cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/ \
        -Dwith-boost=ON \
        -Dwith-ltdl=ON \
        -Dwith-gsl=ON \
        -Dwith-readline=ON \
        -Dwith-python=ON \
        -Dwith-mpi=ON \
        -Dwith-openmp=ON \
        -S /usr/src/nest-simulator \
        -B nest-build && \
  make -j4 -C nest-build && \
  make -C nest-build install

# clean up install/build files
RUN rm -r /usr/src/nest-simulator
RUN rm -r nest-build

# Add NEST binary folder to PATH
RUN echo "source /usr/local/bin/nest_vars.sh" >> root/.bashrc

# ---- install NESTML -----
RUN pip install git+https://github.com/nest/nestml.git@a3a1b0d9fdb26e53aae45d8520c1f5b7dc97eaa3#egg=nestml

# ---- install NEURON ----
RUN git clone --depth 1 -b 8.0.2 https://github.com/neuronsimulator/nrn.git /usr/src/nrn
RUN mkdir nrn-bld

RUN cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/ \
  -DCURSES_NEED_NCURSES=ON \
  -DNRN_ENABLE_INTERVIEWS=OFF \
  -DNRN_ENABLE_MPI=ON \
  -DNRN_ENABLE_RX3D=OFF \
  -DNRN_ENABLE_PYTHON=ON \
  -DNRN_MODULE_INSTALL_OPTIONS="" \
  -S /usr/src/nrn \
  -B nrn-bld

RUN cmake --build nrn-bld --parallel 4 --target install

# clean up
RUN rm -r /usr/src/nrn
RUN rm -r nrn-bld

# ---- pip install some additional things
RUN pip --no-cache-dir install --no-deps pymoo==0.4.2.2
RUN pip --no-cache-dir install --no-deps git+https://github.com/NeuralEnsemble/parameters@b95bac2bd17f03ce600541e435e270a1e1c5a478#egg=parameters

# ---- install LFPykernels (main branch) -----
# RUN pip install git+https://github.com/LFPy/LFPykernels@29d3fcb40101c0bedca775a2985893de312488ad#egg=lfpykernels
RUN pip --no-cache-dir install --no-deps git+https://github.com//alejoe91/MEAutility.git@1.5.1#egg=MEAutility
RUN pip --no-cache-dir install --no-deps git+https://github.com/LFPy/LFPykit.git@v0.4#egg=lfpykit
RUN pip --no-cache-dir install --no-deps git+https://github.com/LFPy/LFPy.git@v2.2.6#egg=LFPy
RUN pip --no-cache-dir install --no-deps git+https://github.com/LFPy/LFPykernels@v0.1rc8#egg=lfpykernels

# install jupyter-lab
RUN pip --no-cache-dir install jupyterlab==3.4.3
