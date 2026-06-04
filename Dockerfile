# Bake the compiled GYSELA compression mini-app into the gyselalibxx runtime
# image, so the benchmark does not need to mount the binary — only the work
# dir. The mini-app binary is statically linked, so the runtime libs from the
# base image are enough.
#
# Build from a gysela-mini-app_io checkout that has been compiled
# (BUILD_COMPRESSION_APP=ON), using that checkout as the build context:
#
#   docker build -f benchmark_gysela/Dockerfile -t gysela-compression \
#       gysela-mini-app_io
#
# The Landau2X2V dataset's launcher (landau_docker_launch.sh) defaults to this
# image and these paths (override with GYSELA_IMAGE / GYSELA_BIN / GYSELA_PDI).
FROM ghcr.io/gyselax/gyselalibxx_env:latest as builder

RUN echo -e '[url "https://github.com/"]\n  insteadOf = git@github.com:' > .gitconfig
RUN git clone --depth 1 https://github.com/gyselax/gysela-mini-app_io.git --recurse-submodules
RUN cd gysela-mini-app_io && cmake -DCMAKE_PREFIX_PATH="/opt/googletest:/opt/openmp/" -B build -S .
RUN cd gysela-mini-app_io && cmake --build build -j 6 compression_app
RUN mkdir -p /opt/gysela
RUN cp gysela-mini-app_io/build/apps/compression/compression_app /opt/gysela/compression_app
RUN cp gysela-mini-app_io/build/apps/compression/pdi_out.yaml /opt/gysela/pdi_out.yaml

FROM ghcr.io/gyselax/gyselalibxx_env:latest
COPY --from=builder /opt/gysela /opt/gysela
