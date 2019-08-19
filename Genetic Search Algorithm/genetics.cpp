/*	Thomas Klimek
	Comp 131 AI
	Assignment #2
*/

/* =============================================================

	To run this:

	g++ genetics.cpp -o genetics
	./genetics


 				Explanation & function contracts
	
	The core of the genetics algorithm follows these steps:
	1. Generate random population
	2. Apply Fitness function to find most fit individuals
	3. Preform cross-overs in population to double the size
	4. Small chance of mutation in each crossover
	5. Cull 50% of the population that is least fit (returning it to original size)
	6. Repeat steps 2-5 for specified number of generations
	7. Return the least fit inividual

	generate random population - create specified number of backpacks, fill them
	with random items.

	fitness function - sort population of backpacks by overall value of items
	inside.

	cross-over - take two parents, a dominant parent and a passive parent.
	Swap one item from the passive parent into the dominant parent that is
	not a duplicate item. This new backpack is the child. The items to be swapped
	are determined randomly.

	mutation - a child experiences a mutation with a 1 in 10 chance. When a child
	is mutated, one of its items is replaced at random with a different default item.

	cull - remove the lest fit 50% of the population.

	Note: The ideal candidate is a backpack with boxes 1, 2, and 6. This backpack has 
	weight: 120 and value: 20. Due to the way that the cross-over and mutation operations
	are defined, we can have trouble reaching this combination under certain circumstances.
	A common problem of local searches is to get stuck at local extrema. The local
	extrema occur when the initial population is not diverse enough. Because neither the
	cross-over or the mutation provide a way of increasing the number of boxes in the
	backpack, if our initial population contains no backpacks with 3 items then this 
	ideal will never be reached. Instead we will find a local maximum of two item backpacks.
	I took a few liberties in this problem to make the implementation more simple and 
	clean, this being one of them. In production code, these issues could be addressed
	by adding items to the backpack during a crossover or mutation with a small probability.
	For the purpose of demonstrating the genetic algorithm, these fixes were not entirely
	necesarry. This also reflects some of the weaknesses of local searches. 

   ============================================================= */


//-----------------------------------------------------------
// includes

#include <stdlib.h>
#include <stdio.h>
#include <iostream> // cout
#include <vector> 	// vector
#include <ctime>	// time	
#include <cstdlib>  // random
#include <algorithm> //algorithm


using namespace std;

//-----------------------------------------------------------
// data types 

/* box struct - represents a box with a weight and value. These 
are the items we are considering adding to the backpack */
typedef struct box {
	int weight, value, key;
	
} box;

/* backpack struct - represents a backpack full of boxes. Has
a vector array of boxes, overall weight and overall value. */
typedef struct backpack {
	vector<box> v;
	int weight;
	int value;
} backpack;

//-----------------------------------------------------------
// global variables

/* Max weight for a backpack */
int MAX_WEIGHT = 120;

/* default boxes */
vector<box> BOXES;
box BOX1;
box BOX2;
box BOX3;
box BOX4;
box BOX5;
box BOX6;
box BOX7;


//-----------------------------------------------------------
// functions

/* initializes the defualt boxes for the problem */
void initialize_boxes(){

	BOX1.weight = 20;
	BOX1.value = 6;
	BOX1.key = 1;
	BOXES.push_back(BOX1);
	BOXES.push_back(BOX1);

	BOX2.weight = 30;
	BOX2.value = 5;
	BOX2.key = 2;
	BOXES.push_back(BOX2);

	BOX3.weight = 60;
	BOX3.value = 8;
	BOX3.key = 3;
	BOXES.push_back(BOX3);

	BOX4.weight = 90;
	BOX4.value = 7;
	BOX4.key = 4;
	BOXES.push_back(BOX4);


	BOX5.weight = 50;
	BOX5.value = 6;
	BOX5.key = 5;
	BOXES.push_back(BOX5);

	BOX6.weight = 70;
	BOX6.value = 9;
	BOX6.key = 6;
	BOXES.push_back(BOX6);

	BOX7.weight = 30;
	BOX7.value = 4;
	BOX7.key = 7;
	BOXES.push_back(BOX7);
}

/* comparison function for backpacks, a helper function used when
sorting backpacks by over all value. This contributes to the fitness
function */
bool compare_backpack(const backpack &a, const backpack &b){
    return a.value > b.value;
}

/* function to print an entire population */
void print_population(vector<backpack> pop){
	int size = pop.size();
	int backpack_size;
	for (int i =0; i < size; i++){
		cout << "backpack " << pop[i].value << ": " << endl;
		backpack_size = pop[i].v.size();
		for (int j = 0; j< backpack_size; j++){
			cout << pop[i].v[j].key << ", " ;
		}
		cout << endl;
	}
}

/* function to print contents of a backpack */
void print_backpack(backpack b){
	int size = b.v.size();
	for (int j = 0; j< size; j++){
			cout << b.v[j].key << ", " ;
		}
		cout << endl;
}

/* function that prints the most fit individual from a sorted population */
void print_best_candidate(vector<backpack> population){
	cout << endl;
	cout << "--Most Fit Individual--" << endl;
	cout << "total weight: " << population[0].weight << ", ";
	cout << "total value: " << population[0].value << endl;
	cout << "items in box: " << endl;
	print_backpack(population[0]);
}

/* returns true if a default item with the key of item
is found in the backpack b and false otherwise */
bool in_backpack(int item, backpack b){
	int size = b.v.size();
	for (int i = 0; i < size; i++){
		if (b.v[i].key == item){
			return true;
		}
	}
	return false;
}

/* fills backpack b with random items */
void randomize_backpack(backpack& b){
	int rand_item;

	while (b.weight <= MAX_WEIGHT){

		rand_item =  (rand() % 7) + 1;
		if (!in_backpack(rand_item, b)){
			if (b.weight + BOXES[rand_item].weight > MAX_WEIGHT) {
				break;
			} else {
				b.v.push_back(BOXES[rand_item]);
				b.weight += BOXES[rand_item].weight;
				b.value += BOXES[rand_item].value;
			}
		}
	}
	
}

/* generates a random population with specified size */
void generate_population(vector<backpack>& pop, int size){
	
	for (int i=0; i<size; i++){
		backpack b;
		b.weight = 0;
		b.value = 0;
		randomize_backpack(b);
		pop.push_back(b);
	}
} 

/* fitness function for a population */
void fitness_function(vector<backpack>& pop){
	/* sorts the population in terms of the most fit individuals */
	sort(pop.begin(), pop.end(), compare_backpack);
}

/* mutates a backpack b by randomly changing one of its items with a default item */
void mutate(backpack& b){
	int size = b.v.size();
	int random_child_trait;
	int random_mutation;

	/* chose a random item in our child backpack, and a random default item.
	make sure the random default item is not in the backpack to avoid duplicates */
	do {
		random_child_trait = (rand() % size);
		random_mutation = (rand() % 7) + 1;
	} while ((in_backpack(random_mutation, b)));

	/* subtract the value and weight of the item removed from the child */
	b.value -= b.v[random_child_trait].value;
	b.weight -= b.v[random_child_trait].weight;

	b.v[random_child_trait] = BOXES[random_mutation];

	/* add the value and weight of the new item */
	b.value += b.v[random_child_trait].value;
	b.weight += b.v[random_child_trait].weight;

	/* if the child's weight exceeds the max weight, it has no value */
	if (b.weight > MAX_WEIGHT){
		b.value = 0;
	}
}

/* crossover function, it takes a child which is initially equivalent to one parent
and a second parent. Preforms a random crossover of items from the second parent to
the child */
void crossover(backpack& p2, backpack& child){
	int random_parent_trait;
	int random_child_trait;
	int parent_size = p2.v.size();
	int child_size = child.v.size();
	
	/* generate random numbers with the condition that you do not
		add the same number to the backpack twice */
	do {
		random_child_trait = (rand() % child_size);
		random_parent_trait = (rand() % parent_size);
	} while ((in_backpack(p2.v[random_parent_trait].key, child)) and 
				(p2.v[random_parent_trait].key != child.v[random_child_trait].key));
	
	/* subtract the value and weight of the item removed from the child */
	child.value -= child.v[random_child_trait].value;
	child.weight -= child.v[random_child_trait].weight;

	/* swap items between child and parent */
	child.v[random_child_trait] = p2.v[random_parent_trait];

	/* add value and weight of new item to child*/
	child.value += child.v[random_child_trait].value;
	child.weight += child.v[random_child_trait].weight;

	/* if the child's weight exceeds the max weight, it has no value */
	if (child.weight > MAX_WEIGHT){
		child.value = 0;
	}

}

/* grows our population by preforming a crossover between each member and another,
random member of the population. The child of these two parents has a small chance
of mutation */
void grow_population(vector<backpack>& pop){

	int random_parent;
	int mutation_percentage = 10; // 1 in 10 percent chance of a mutation
	int mutation;
	int size = pop.size();

	for (int i = 0; i < size; i++){
		/* make new child from each individual in the population */
		backpack child = pop[i];
		
		// generate a random parent different from the first parent i
		random_parent = ((rand() % (size/2)) + i) % (size/2);

		/* cross over between the two parents to create a unique child */
		crossover(pop[random_parent], child);
		

		/* small percentage of a mutation to the child */
		mutation = (rand() % mutation_percentage) + 1;
		if (mutation == 1){
			mutate(child);
		}

		/* add the child to the population */
		pop.push_back(child);
	}

}

/* culls the least fit 50% of our population */
void cull(vector<backpack>& pop){
	int size = pop.size();
	sort(pop.begin(), pop.end(), compare_backpack);
	pop.erase (pop.begin()+(size/2),pop.begin()+size);
}

/* overall genetic search algorithm implementation */
void genetic_search(int generations, int population_size, char population_log){
	vector<backpack> population;
	generate_population(population, population_size);

	for (int i = 0; i < generations; i++){
		
		fitness_function(population);
		grow_population(population);
		cull(population);
		
		if (population_log == 'y'){
			cout << "generation " << i << endl;
			cout << endl;
			print_population(population);
		}
	}
	print_best_candidate(population);

}

//-----------------------------------------------------------
// main
int main(){
	srand(time(NULL));
	initialize_boxes();
	
	char population_log;
	int generations;
	int population_size;


	cout << endl;
	cout << " ----------------------------------------------------------------- " << endl;
	cout << "Welcome the genetic search algorithm. Here we will try to solve the" << endl;
	cout << "backpack problem, by using a genetic search and adopting a natural " << endl;
	cout << "selection analogy. Written By - Thomas Klimek" << endl;
	cout << " ----------------------------------------------------------------- " << endl;
	cout << endl;

	cout << "Please enter the size of the population for the search: " << endl;
	cin >> population_size;
	cout << "Please enter the size of the number of generations to simulate: " << endl;
	cin >> generations;
	cout << "would you like a log of the population through generations? y/n" << endl;
	cin >> population_log;


	genetic_search(generations, population_size, population_log);

	return 0;
}