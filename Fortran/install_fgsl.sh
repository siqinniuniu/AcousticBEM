export PKG_CONFIG_PATH=`pwd`/gsl/lib/pkgconfig
export gsl_LIBS=`pwd`/gsl
[ -f ./fgsl.tar.gz ] && echo "No need to download fgsl" || wget https://github.com/reinh-bader/fgsl/archive/v1.2.0.tar.gz -O fgsl.tar.gz
mkdir -p fgsl_build
cd fgsl_build
tar -xzf ../fgsl.tar.gz --strip 1
autoreconf -fi
./configure CFLAGS="-Wall" FCFLAGS="-Wall" --prefix=${gsl_LIBS}
make
make check
make install
cd ..
rm -rf fgsl_build