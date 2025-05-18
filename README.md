# qrm~ 
Near Realtime Resonant Model Generation

qrm~ (quick resonant model) is an object for Max intended for near real time generation of resonance models from signals in buffer~ objects. It uses a method of fractional bin analysis to produce resonant models suitable for use with CNMAT's sinusoids~ and resonators~ objects. The object is the subject of a 2025 ICMC paper and is still considered an experimental object as of May 2025.

# Building
This software has dependencies in [FFTW](https://www.fftw.org/). Sources should be compiled according to the instructions in the [CNMAT-Externs](https://github.com/CNMAT/CNMAT-Externs) repo.  Once those sources are compiled, you should be able to build from the .xcodeproject in the /build directory. 
