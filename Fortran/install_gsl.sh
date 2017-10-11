export gsl_INSTALL_DIR=`pwd`/gsl
[ -f ./gsl.tar.gz ] && echo "No need to download gsl" || wget http://ftp.igh.cnrs.fr/pub/gnu/gsl/gsl-2.3.tar.gz -O gsl.tar.gz
mkdir -p gsl_build
cd gsl_build
tar -xzf ../gsl.tar.gz --strip 1
autoreconf -fi
./configure CFLAGS="-Wall" --prefix=${gsl_INSTALL_DIR}
make > /dev/null
make install > /dev/null
cd ..
rm -rf gsl_build