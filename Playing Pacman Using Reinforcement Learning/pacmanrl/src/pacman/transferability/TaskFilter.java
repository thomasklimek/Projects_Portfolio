package pacman.transferability;

import java.util.ArrayList;
import java.util.Random;

public class TaskFilter {

	public static ArrayList<Task[]> getPairs(ArrayList<Task> tasks_a, ArrayList<Task> tasks_b ){
		ArrayList<Task[]> pairs = new ArrayList<Task[]>();
		for (int i = 0; i < tasks_a.size(); i ++){
			for (int j = 0; j < tasks_b.size(); j++){
				
				Task [] pair = new Task[2];
				pair[0]=tasks_a.get(i);
				pair[1]=tasks_b.get(j);
				pairs.add(pair);
				
			}
		}
		
		return pairs;
	}
	
	public static ArrayList<Task[]> getPairs(ArrayList<Task> tasks){
		ArrayList<Task[]> pairs = new ArrayList<Task[]>();
		for (int i = 0; i < tasks.size(); i ++){
			for (int j = 0; j < tasks.size(); j++){
				if (i != j){
					Task [] pair = new Task[2];
					pair[0]=tasks.get(i);
					pair[1]=tasks.get(j);
					pairs.add(pair);
				}
			}
		}
		
		return pairs;
	}
	
	public static ArrayList<Task> getRandomSubset(ArrayList<Task> tasks, int target, Random r){
		 ArrayList<Task> results = new  ArrayList<Task>();
		 ArrayList<Task> temp_copy = new  ArrayList<Task>(tasks);
		 
		
		 
		 for (int i = 0; i < target; i ++){
			 int r_next = r.nextInt(temp_copy.size());
			 results.add(temp_copy.get(r_next));
			 temp_copy.remove(r_next);
		 }
		 
		 
		 return results;
	}
	
	public static ArrayList<Task> getRandomSubset(ArrayList<Task> tasks, double fraction, Random r){
		 ArrayList<Task> results = new  ArrayList<Task>();
		 ArrayList<Task> temp_copy = new  ArrayList<Task>(tasks);
		 
		 int target = (int)Math.floor(fraction*(double)temp_copy.size());
		 
		 for (int i = 0; i < target; i ++){
			 int r_next = r.nextInt(temp_copy.size());
			 results.add(temp_copy.get(r_next));
			 temp_copy.remove(r_next);
		 }
		 
		 
		 return results;
	}
	
	public static ArrayList<Task> getTrainFold(ArrayList<Task> tasks, int num_folds, int current_fold){
		 ArrayList<Task> results = new  ArrayList<Task>();
		 
		 for (int i = 0; i < tasks.size(); i++){
			 if (i % num_folds != current_fold){
				 results.add(tasks.get(i));
			 }
		 }
		 
		 return results;
	}
	
	public static ArrayList<Task> getTestFold(ArrayList<Task> tasks, int num_folds, int current_fold){
		 ArrayList<Task> results = new  ArrayList<Task>();
		 
		 for (int i = 0; i < tasks.size(); i++){
			 if (i % num_folds == current_fold){
				 results.add(tasks.get(i));
			 }
		 }
		 
		 return results;
	}
	
	
	public static ArrayList<Task> filterByMaze(ArrayList<Task> tasks, int mazenum){
		ArrayList<Task> results = new ArrayList<Task>();
		
		
		
		return results;
	}
}
