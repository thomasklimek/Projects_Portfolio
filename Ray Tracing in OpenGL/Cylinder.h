#ifndef CYLINDER_H
#define CYLINDER_H

#include "Shape.h"
#include <vector>
#include <stdio.h>
#include <iostream>

using namespace std;

class Cylinder : public Shape {
public:
	Cylinder() {lastX = -1;lastY = -1;};
	~Cylinder() {};

	int lastX;
	int lastY;

	OBJ_TYPE getType() {
		return SHAPE_CYLINDER;
	}

	vector<vector<float> > triangles;
	vector<vector<float> > normals;
	
	void draw() {
		if (lastX != m_segmentsX || lastY != m_segmentsY){
			lastX = m_segmentsX;
			lastY = m_segmentsY;
			float radiansPerSlice = (2 * M_PI) / m_segmentsX;

			triangles.clear();
			normals.clear();

			vector<float> centerTop;
			centerTop.push_back(0.0f);
			centerTop.push_back(0.5f);
			centerTop.push_back(0.0f);

			vector<float> centerBottom;
			centerBottom.push_back(0.0f);
			centerBottom.push_back(-0.5f);
			centerBottom.push_back(0.0f);

			// top and bottom
			for (int n = 0; n < 2; n++){
				for (int slice = 0; slice < m_segmentsX; slice++){
					vector<float> p1, p2;

					float x1 = cos(slice * radiansPerSlice) / 2.0f;
					float y1 = 0.5f - n;
					float z1 = sin(slice * radiansPerSlice) / 2.0f;

					float x2 = cos((slice + 1) * radiansPerSlice) / 2.0f;
					float y2 = 0.5f - n;
					float z2 = sin((slice + 1) * radiansPerSlice) / 2.0f;

					p1.push_back(x1);
					p1.push_back(y1);
					p1.push_back(z1);

					p2.push_back(x2);
					p2.push_back(y2);
					p2.push_back(z2);

					if (n == 1){
						triangles.push_back(p1);
						triangles.push_back(p2);
						triangles.push_back(centerBottom);

						vector<float> downV;
						downV.push_back(0.0f);
						downV.push_back(-1.0f);
						downV.push_back(0.0f);

						normals.push_back(downV);
						normals.push_back(downV);
						normals.push_back(downV);
					}
					else{
						triangles.push_back(centerTop);
						triangles.push_back(p2);
						triangles.push_back(p1);

						vector<float> upV;
						upV.push_back(0.0f);
						upV.push_back(1.0f);
						upV.push_back(0.0f);

						normals.push_back(upV);
						normals.push_back(upV);
						normals.push_back(upV);
					}
				}
			}

			float yLength = (1.0f / m_segmentsY);
			//sides of cylinder
			for (int slice = 0; slice < m_segmentsX; slice++){
				float x1 = cos(slice * radiansPerSlice) / 2.0f;
				float x2 = cos((slice + 1) * radiansPerSlice) / 2.0f;

				float z1 = sin(slice * radiansPerSlice) / 2.0f;
				float z2 = sin((slice + 1) * radiansPerSlice) / 2.0f;
				for (int y = 0; y < m_segmentsY; y++){
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;

					vector<float> p1, p2, p3, p4;

					p1.push_back(x1);
					p1.push_back(y1);
					p1.push_back(z1);

					p2.push_back(x2);
					p2.push_back(y1);
					p2.push_back(z2);

					p3.push_back(x1);
					p3.push_back(y2);
					p3.push_back(z1);

					p4.push_back(x2);
					p4.push_back(y2);
					p4.push_back(z2);

					vector<float> n1, n2, n3, n4;
					n1.push_back(x1 * 2.0f);
					n1.push_back(0.0f);
					n1.push_back(z1 * 2.0f);

					n2.push_back(x2 * 2.0f);
					n2.push_back(0.0f);
					n2.push_back(z2 * 2.0f);

					n3.push_back(x1 * 2.0f);
					n3.push_back(0.0f);
					n3.push_back(z1 * 2.0f);

					n4.push_back(x2 * 2.0f);
					n4.push_back(0.0f);
					n4.push_back(z2 * 2.0f);

					triangles.push_back(p2);
					normals.push_back(n2);

					triangles.push_back(p1);
					normals.push_back(n1);

					triangles.push_back(p3);
					normals.push_back(n3);
					


					triangles.push_back(p4);
					normals.push_back(n4);

					triangles.push_back(p2);
					normals.push_back(n2);

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