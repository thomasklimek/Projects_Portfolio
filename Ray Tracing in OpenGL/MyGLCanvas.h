#pragma once

#ifndef MYGLCANVAS_H
#define MYGLCANVAS_H

#include <FL/gl.h>
#include <FL/glut.h>
#include <FL/glu.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <time.h>
#include <iostream>

#include "Shape.h"
#include "Cube.h"
#include "Cylinder.h"
#include "Cone.h"
#include "Sphere.h"

#include "Camera.h"
#include "scene/SceneParser.h"

class MyGLCanvas : public Fl_Gl_Window {
public:
	glm::vec3 rotVec;
	glm::vec3 eyePosition;
	GLubyte* pixels = NULL;

	int isectOnly;
	int segmentsX, segmentsY;
	int recursion_val;
	float scale;

	OBJ_TYPE objType;
	Cube* cube;
	Cylinder* cylinder;
	Cone* cone;
	Sphere* sphere;
	Shape* shape;

	Camera* camera;
	SceneParser* parser;

	MyGLCanvas(int x, int y, int w, int h, const char *l = 0);
	~MyGLCanvas();
	void renderShape(OBJ_TYPE type);
	void setSegments();
	void loadSceneFile(const char* filenamePath);
	void renderScene();
	glm::vec3 generateRay(int pixelX, int pixelY, glm::mat4 camToWorld);
	double intersect(glm::vec3 eyePointP, glm::vec3 rayV, glm::mat4 transformMatrix, OBJ_TYPE type);
	SceneColor getColor(SceneMaterial prim, glm::vec3 normal, glm::vec3 intersection, glm::vec3 ray, RenderedPrimitive p, float t);
	glm::vec3 getUnit(glm::vec3 eye, glm::vec3 ray, RenderedPrimitive p, float t);
	SceneColor recursiveRayTrace(glm::vec3 eyePointP, glm::vec3 rayV, int recursion);
	float getNearestIntersect(glm::vec3 rayV, glm::vec3 eyePointP);

private:
	void setpixel(GLubyte* buf, int x, int y, int r, int g, int b);

	void draw();

	int handle(int);
	void resize(int x, int y, int w, int h);
	void updateCamera(int width, int height);
};

#endif // !MYGLCANVAS_H