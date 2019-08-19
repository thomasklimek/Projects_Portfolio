#Makefile for a4 

build:
	clang++ `fltk-config --use-gl --ldflags` -O2 MyGLCanvas.cpp ./scene/SceneParser.cpp ./scene/tinyxmlparser.cpp ./scene/tinyxmlerror.cpp ./scene/tinyxml.cpp ./scene/tinystr.cpp ./scene/ppm.cpp Camera.cpp main.cpp -o a5

clean:
	rm a5 

