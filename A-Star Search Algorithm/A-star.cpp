/*	Thomas Klimek
	Comp 131 AI
	Assignment #1
*/
#include <stdlib.h>
#include <stdio.h>
#include <iostream>       // std::cout
#include <queue>          // std::priority_queue
#include <vector>         // std::vector

/* ====================================================================
 *					Pancake Problem solved with A* Search
 *
 * -- to run code "g++ A-star.cpp -o A-star ./A-star "
 *	
 * --------------------------------------------------------------------			   
 * Explanation:
 * This implementation is based around the following solution for the pancake 
 * problem. 
 *		1. Find the largest unordered pancake
 *		2. Flip it to the top of the stack
 *		3. Flip it from the top to its correct position
 *		4. Repeat for the rest of the pancakes
 *
 * Overall Structure: The program utilizes the A* search to sort a stack of
 * pancakes. This A* search is implemented in the main function. 
 * 
 * How it works is the A* is given the initial state of the stack of pancakes
 * as a node. From here it initializes a priority queue with the initial node.
 * The priority for the queue is g(n) + h(n) where g and h are heuristic 
 * functions for forward cost and backward cost, respectively. Forward cost
 * represents how many flips it will take to correctly position the next 
 * pancake. Backward cost represents how many pancakes are already correctly
 * positioned. When the A* algorithm expands on a node, it preforms every
 * possible flip on the stack, and adds these configurations to the frontier
 * as nodes. Then it chooses the node with the most priority (min g(n) + h(n))
 * It goal tests this node, and continues to expand upon it until it reaches a
 * goal state. Each flip is stored in a simple queue, and the order of flips
 * is printed at the end.
 *
 *


/* This is a node in our A* search. it represents a configuration of a stack
of pancakes in an array. The left most side is the bottow of the stack and the
right most is the top */
typedef struct node {
	/* priority determined from forward and backward cost */
	int priority;
	/* configuration of pancakes */
	int array[5];
	/* comparison operator for priority queue. minimum of cost function*/
	bool operator<(const node& rhs) const
    {
        return priority > rhs.priority;
    }
} node;

/* This function checks to see if a node contains a goal state i.e. 
the pancake stack is sorted */
bool goalstate(node n){
	for (int i=0; i<5; i++){
		if (n.array[i] != 5 - i)
			return false;
	}
	return true;
}

/* This function determines the cost of a stack of pancakes. It uses
two heuristics to determine backward cost and forward cost and returns
the sum */
int priority_function(int array[]){
	// forward cost
	int forward_cost=0;

	/* find the next pancake that needs to be flipped */
	int nextflip;
	for (int i = 0; i<5; i++){
		if (array[i] != 5-i){

			nextflip = 5-i;
			break;
		}
	}
	/* if the last member of the array is the next pancake to be flipped
	the forward cost is 1 because it will require 1 flip to position it 
	otherwise 2 flips are sufficient for positioning it*/
	if (array[4] == nextflip){
		forward_cost = 1;
	} else {
		forward_cost = 2;
	}

	//backward cost
	int backward_cost;
	int prefix = 0;
	for (int i=0; i<5; i++){
		if (array[i] == 5-i){
			prefix++;
		} else{
			break;
		}
	}
	backward_cost = 5 - prefix;

	// return total cost which functions as priority in priority queue
	return backward_cost + forward_cost;

}

/* Function for flipping a pancake, takes index i and array of pancakes. 
insert spatula at index i and flip all above */
void flip(int arr[], int i)
{
    int temp, start = 4;
    while (start > i)
    {
        temp = arr[start];
        arr[start] = arr[i];
        arr[i] = temp;
        start--;
        i++;
    }

}
/* node expansion function -- adds all children of a node n to the frontier.
children here are defined as a stack of pancakes that can be achieved from one
flip of the stack n */
void expandNode(node n, std::priority_queue<node>& frontier){
	/* generate children. Children of a pancake stack are all possible 
	configurations after one flip */
	for (int i = 0; i< 4; i++){
		node child = n;
		// child array by performing flip
		flip(child.array, i);
		// child priority from heuristics
		child.priority = priority_function(child.array);
		// add child to frontier
		frontier.push(child);
	}
}


/* basic array printing function */
void printArray(int arr[])
{
    for (int i = 0; i < 5; ++i)
        printf("%d ", arr[i]);
}
 

int main(){

	/* Initialize queue and get first pancake */
	node pancake_stack;
	std::priority_queue<node> frontier;
	std::queue<node> flips;

	/* Initialize first pancake, default initializer and cin code for user defined stacks */
	/* pancake stack [1, 5, 3, 2, 4] */
	pancake_stack.array[0] = 1;
	pancake_stack.array[1] = 5;
	pancake_stack.array[2] = 3;
	pancake_stack.array[3] = 4;
	pancake_stack.array[4] = 2;

	/* User defined stacks */
	std::cout << "Please enter a pancake stack in the following form: 1 2 3 4 5" << std::endl;
	for (int i =0; i< 5; i++){
		std::cin >> pancake_stack.array[i];
	}
	pancake_stack.priority = priority_function(pancake_stack.array);

	/* A* Algorithm for Sorting pancakes 
	 * ==================================================*/
	frontier.push(pancake_stack);

	node current;
	do{
		if (frontier.empty()){
			std::cout << "FAILURE" << std::endl;
		}
		
		current = frontier.top();
		frontier.pop();

		flips.push(current);

		
		if (goalstate(current)){
			break;
		}
		expandNode(current, frontier);

	}while (true);

	 /* ==================================================*/

	/* Printing procedure to return results */
	int count = 0;
	while (!flips.empty()){
			current = flips.front();
			flips.pop();
			if (count == 0){
				std::cout << "Initial State:" << std::endl;
				printArray(current.array);
				std::cout << std::endl;
				std::cout << std::endl;
			} else {
				std::cout << "flip #" << count << std::endl;
				printArray(current.array);
				std::cout << std::endl;
				std::cout << std::endl;
			}

			count++;
		}



	return 0;
}