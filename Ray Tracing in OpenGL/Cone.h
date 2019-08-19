#ifndef CONE_H
#define CONE_H

#include "Shape.h"

class Cone : public Shape {
public:
	Cone() {lastX = -1;lastY = -1;};
	~Cone() {};
	int lastX;
	int lastY;

	OBJ_TYPE getType() {
		return SHAPE_CONE;
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
			for (int slice = 0; slice < m_segmentsX; slice++){
				vector<float> p1, p2;

				float x1 = cos(slice * radiansPerSlice) / 2.0f;
				float y1 = -0.5f;
				float z1 = sin(slice * radiansPerSlice) / 2.0f;

				float x2 = cos((slice + 1) * radiansPerSlice) / 2.0f;
				float y2 = -0.5f;
				float z2 = sin((slice + 1) * radiansPerSlice) / 2.0f;

				p1.push_back(x1);
				p1.push_back(y1);
				p1.push_back(z1);

				p2.push_back(x2);
				p2.push_back(y2);
				p2.push_back(z2);

				
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

			float yLength = (1.0f / m_segmentsY);
			//sides
			for (int slice = 0; slice < m_segmentsX; slice++){
				float x1 = cos(slice * radiansPerSlice) / 2.0f;
				float x2 = cos((slice + 1) * radiansPerSlice) / 2.0f;

				float z1 = sin(slice * radiansPerSlice) / 2.0f;
				float z2 = sin((slice + 1) * radiansPerSlice) / 2.0f;
				for (int y = 0; y < m_segmentsY; y++){
					float y1 = y * yLength - 0.5f;
					float y2 = (y + 1) * yLength - 0.5f;

					float h1 = y1 + 0.5f;
					float h2 = y2 + 0.5f;


					vector<float> p1, p2, p3, p4;

					p1.push_back(x1 * (1.0f - h1));
					p1.push_back(y1);
					p1.push_back(z1 * (1.0f - h1));

					p2.push_back(x2 * (1.0f - h1));
					p2.push_back(y1);
					p2.push_back(z2 * (1.0f - h1));

					p3.push_back(x1 * (1.0f - h2));
					p3.push_back(y2);
					p3.push_back(z1 * (1.0f - h2));

					p4.push_back(x2 * (1.0f - h2));
					p4.push_back(y2);
					p4.push_back(z2 * (1.0f - h2));

					vector<float> n1, n2, n3, n4;
					

					float n1y = 0.5f * sqrt(x1*x1 + z1*z1);
					float n2y = 0.5f * sqrt(x2*x2 + z2*z2);
					float n3y = 0.5f * sqrt(x1*x1 + z1*z1);
					float n4y = 0.5f * sqrt(x2*x2 + z2*z2);

					n1.push_back(x1 / sqrt(x1*x1 + n1y*n1y + z1*z1));
					n1.push_back(n1y / sqrt(x1*x1 + n1y*n1y + z1*z1));
					n1.push_back(z1 / sqrt(x1*x1 + n1y*n1y + z1*z1));

					n2.push_back(x2 / sqrt(x2*x2 + n2y*n2y + z2*z2));
					n2.push_back(n2y / sqrt(x2*x2 + n2y*n2y + z2*z2));
					n2.push_back(z2 / sqrt(x2*x2 + n2y*n2y + z2*z2));

					n3.push_back(x1 / sqrt(x1*x1 + n3y*n3y + z1*z1));
					n3.push_back(n3y / sqrt(x1*x1 + n3y*n3y + z1*z1));
					n3.push_back(z1 / sqrt(x1*x1 + n3y*n3y + z1*z1));

					n4.push_back(x2 / sqrt(x2*x2 + n4y*n4y + z2*z2));
					n4.push_back(n4y / sqrt(x2*x2 + n4y*n4y + z2*z2));
					n4.push_back(z2 / sqrt(x2*x2 + n4y*n4y + z2*z2));

					
					triangles.push_back(p1);
					normals.push_back(n1);
					triangles.push_back(p3);
					normals.push_back(n3);
					triangles.push_back(p2);
					normals.push_back(n2);
					

					triangles.push_back(p3);
					normals.push_back(n3);
					triangles.push_back(p4);
					normals.push_back(n4);
					triangles.push_back(p2);
					normals.push_back(n2);
					
					
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