#ifndef SPHERE_H
#define SPHERE_H

#include "Shape.h"
#include <vector>

using namespace std;

class Sphere : public Shape {
public:
	Sphere() {lastX = -1;lastY = -1;};
	~Sphere() {};

	int lastX;
	int lastY;

	vector<vector<float> > triangles;
	vector<vector<float> > normals;

	OBJ_TYPE getType() {
		return SHAPE_SPHERE;
	}

	float getX(float theta, float phi){
		return 0.5f * sin(phi) * cos(theta);
	}

	float getY(float theta, float phi){
		return 0.5f * cos(phi);
	}

	float getZ(float theta, float phi){
		return 0.5f * sin(phi) * sin(theta);
	}

	// normal vector is just the point of the sphere, normalized
	vector<float> calculateNormal(vector<float> p){
		vector<float> normal;
		float magnitude = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
		normal.push_back(p[0] / magnitude);
		normal.push_back(p[1] / magnitude);
		normal.push_back(p[2] / magnitude);
		return normal;
	}

	void draw() {
		if (lastX != m_segmentsX || lastY != m_segmentsY){
			lastX = m_segmentsX;
			lastY = m_segmentsY;
			float radiansPerSliceTheta = (2 * M_PI) / m_segmentsX;
			float radiansPerSlicePhi = (M_PI) / m_segmentsY;


			triangles.clear();
			normals.clear();

			//sphere
			for (int slice = 0; slice < m_segmentsX; slice++){
				for (int longitudinal = 0; longitudinal < m_segmentsY; longitudinal++){

					float theta = slice * radiansPerSliceTheta;
					float phi = longitudinal * radiansPerSlicePhi;

					float theta2 = (slice + 1) * radiansPerSliceTheta;
					float phi2 = (longitudinal + 1) * radiansPerSlicePhi;

					

					vector<float> p1, p2, p3, p4;

					p1.push_back(getX(theta, phi));
					p1.push_back(getY(theta, phi));
					p1.push_back(getZ(theta, phi));

					p2.push_back(getX(theta2, phi));
					p2.push_back(getY(theta2, phi));
					p2.push_back(getZ(theta2, phi));

					p3.push_back(getX(theta, phi2));
					p3.push_back(getY(theta, phi2));
					p3.push_back(getZ(theta, phi2));

					p4.push_back(getX(theta2, phi2));
					p4.push_back(getY(theta2, phi2));
					p4.push_back(getZ(theta2, phi2));

					vector<float> n1 = calculateNormal(p1);
					vector<float> n2 = calculateNormal(p2);
					vector<float> n3 = calculateNormal(p3);
					vector<float> n4 = calculateNormal(p4);

					triangles.push_back(p1);
					normals.push_back(n1);
					triangles.push_back(p2);
					normals.push_back(n2);
					triangles.push_back(p3);
					normals.push_back(n3);
					

					triangles.push_back(p2);
					normals.push_back(n2);
					triangles.push_back(p4);
					normals.push_back(n4);
					triangles.push_back(p3);
					normals.push_back(n3);

					
				}

			}
		}

		glBegin(GL_TRIANGLES);
		for (int i = 0; i < triangles.size(); i++){
			glNormal3f(normals[i][0], normals[i][1], normals[i][2]);
			glVertex3f(triangles[i][0], triangles[i][1], triangles[i][2]);
		}
		glEnd();

	};

	void drawNormal() {
		glLineWidth(2); 
		glColor3f(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
		for (int i = 0; i < triangles.size(); i++){
			glVertex3f(triangles[i][0], triangles[i][1], triangles[i][2]);

			glVertex3f(triangles[i][0] + normals[i][0] * .1f, 
					   triangles[i][1] + normals[i][1] * .1f, 
					   triangles[i][2] + normals[i][2] * .1f);
		}
		glEnd();
	};

private:
};

#endif