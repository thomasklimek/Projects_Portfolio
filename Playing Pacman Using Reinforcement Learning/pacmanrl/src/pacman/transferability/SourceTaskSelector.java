package pacman.transferability;

import java.util.ArrayList;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class SourceTaskSelector {

	InstancesCreatorTL IC;
	
	Classifier TL_model;
	
	public SourceTaskSelector(InstancesCreatorTL I){
		IC = I;
	}
	
	public void setClassifier(Classifier C_in){
		try {
			TL_model = AbstractClassifier.makeCopy(C_in);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void trainWithPairs(ArrayList<Task[]> pairs){
		//create instances for all pairs
		Instances weka_data = IC.createInstances(pairs);
						
		//train
		try {
			TL_model.buildClassifier(weka_data);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public void train(ArrayList<Task> tasks){
		//compute all pairs
		//create all pairs
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

		this.trainWithPairs(pairs);
	}
	
	public double [] getScores(ArrayList<Task> options, Task target){
		double [] score = new double[options.size()];
		for (int i = 0; i < options.size(); i ++){
			score[i] = this.processPair(options.get(i), target);
		}
		
		return score;
	}
	
	public  ArrayList<Task> sortSources(ArrayList<Task> options, Task target){
		
		double [] score = new double[options.size()];
		for (int i = 0; i < options.size(); i ++){
			score[i] = this.processPair(options.get(i), target);
		}
			
		Task [] tasks = new Task[options.size()];
		for (int i = 0; i < options.size(); i ++){
			tasks[i]=options.get(i);
		}
		
		int n = score.length;
		double tempScore;
		
		Task tempTask;
		 
		for (int c = 0; c < ( n - 1 ); c++) {
			for (int d = 0; d < n - c - 1; d++) {
		       if (score[d] < score[d+1]) /* For descending order use < */
		       {
		        	
		    	   tempScore = score[d];
		        	tempTask = tasks[d];
		        	
		        	score[d]   = score[d+1];
		        	tasks[d]=tasks[d+1];
		          
		        	score[d+1]=tempScore;
		        	tasks[d+1]=tempTask;
		        	
		        }
		     }
		 }
		 
		 ArrayList<Task> sorted = new ArrayList<Task>();
		 for (int i = 0; i < tasks.length; i++)
			 sorted.add(tasks[i]);
		 
		 return sorted;
		
		
		
	}
	
	public Task findBestSource(ArrayList<Task> options, Task target){
		int best_index = -1;
		double best_score = -10000.0;/*Double.MIN_VALUE;*/
		
		for (int i = 0; i < options.size(); i ++){
			double score_i = this.processPair(options.get(i), target);
			
			if (score_i > best_score){
				best_score = score_i;
				best_index = i;
			}
		}
		
		return options.get(best_index);
	}
	
	public double processPair(Task task_from, Task task_to){
		
		
		Instance X = IC.createInstance(task_from, task_to);
		
		try {
			double result = TL_model.classifyInstance(X);
			return result;
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return 0.0;
	}
	
	
	public Classifier getClassifier(){
		return TL_model;
	}
}
