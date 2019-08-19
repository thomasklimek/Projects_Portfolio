package pacman.transferability;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class UtilsTL {
	
	public static ArrayList<Task> shuffleTasks(ArrayList<Task> tasks, Random r){
		 ArrayList<Task> original = new ArrayList<Task>(tasks);
		 
		 ArrayList<Task> shuffled = new ArrayList<Task>();
		 
		 for (int i = 0; i < tasks.size(); i++){
			 int next = r.nextInt(original.size());
			 shuffled.add(original.get(next));
			 original.remove(next);
		 }
		 
		 return shuffled;
	}
	
	public static void saveArray(double [] hist, String filename){
		try {
			FileWriter FW = new FileWriter(new File(filename));
			
			for (int i = 0; i < hist.length; i++){
				FW.write(new String(hist[i]+"\n"));
			}
			
			FW.close();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static double [] toDoubleArray(ArrayList<Double> input){
		double [] result = new double[input.size()];
		for (int i= 0; i < input.size(); i++){
			result[i]=input.get(i).doubleValue();
		}
		return result;
	}
	
	public static double [] toIntegerArray(ArrayList<Integer> input){
		double [] result = new double[input.size()];
		for (int i= 0; i < input.size(); i++){
			result[i]=input.get(i).intValue();
		}
		return result;
	}
	
	
	
	public static double computeDiscountedCumulativeGain(double [] rel, int k){
		double r = 0;
		
		r = rel[0];
		
		for (int i = 1; i < k; i++){
			r += (rel[i])/(Math.log(i+1));
		}
		
		return r;
		
	}
	
	public static ArrayList<Task> sortTasksByDistance(ArrayList<Task> options, Task target){
		
		double [] distances = new double[options.size()];
		for (int i = 0; i < options.size(); i ++){
			distances[i] = getTaskDistance(options.get(i),target);
		}
			
		Task [] tasks = new Task[options.size()];
		for (int i = 0; i < options.size(); i ++){
			tasks[i]=options.get(i);
		}
		
		int n = distances.length;
		double tempScore;
		
		Task tempTask;
		 
		for (int c = 0; c < ( n - 1 ); c++) {
			for (int d = 0; d < n - c - 1; d++) {
		       if (distances[d] > distances[d+1]) /* For descending order use < */
		       {
		        	
		    	   tempScore = distances[d];
		        	tempTask = tasks[d];
		        	
		        	distances[d]   = distances[d+1];
		        	tasks[d]=tasks[d+1];
		          
		        	distances[d+1]=tempScore;
		        	tasks[d+1]=tempTask;
		        	
		        }
		     }
		 }
		 
		 ArrayList<Task> sorted = new ArrayList<Task>();
		 for (int i = 0; i < tasks.length; i++)
			 sorted.add(tasks[i]);
		 
		 return sorted;
		
		
		
	}
	
	public static Task getClosestTask(ArrayList<Task> options, Task target){
		int best_index = -1;
		double best_score = Double.MAX_VALUE;/*Double.MIN_VALUE;*/
		
		for (int i = 0; i < options.size(); i ++){
			double distance_i = getTaskDistance(options.get(i),target);
			
			if (distance_i < best_score){
				best_score = distance_i;
				best_index = i;
			}
		}
		
		return options.get(best_index);
	}
	
	public static double getTaskDistance(Task A, Task B){
		double d = 0;
		
		ArrayList<String> types = A.getAttributeTypes();
		
		for (int i = 0; i < types.size(); i ++){
			double v_ai = A.getAttributeValues().get(i).doubleValue();
			double v_bi = B.getAttributeValues().get(i).doubleValue();
			
			if (types.get(i).equals("numeric")){
				d += Math.sqrt( (v_ai-v_bi)*(v_ai-v_bi));
			}
			else {
				if (v_ai != v_bi){
					d+=1.0;
				}
			}
		}
		
		
		return d;
	}
}
