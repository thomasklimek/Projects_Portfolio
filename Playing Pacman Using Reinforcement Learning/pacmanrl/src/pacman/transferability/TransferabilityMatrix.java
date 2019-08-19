package pacman.transferability;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

public class TransferabilityMatrix {

	ArrayList<String> tasks;
	
	//the entry at [i][j] contains the measure for transfering
	//from task j to task i
	double [][] matrix;
	
	public TransferabilityMatrix(ArrayList<String> tasks){
		this.tasks = tasks;
		matrix = new double[tasks.size()][tasks.size()];
	}
	
	
	public double [] getScores(ArrayList<Task> options, Task target){
		double [] score = new double[options.size()];
		for (int i = 0; i < options.size(); i ++){
			score[i] = this.getValue(options.get(i).getName(), target.getName());
		}
		
		return score;
	}
	
	public ArrayList<Task> sortTasksByScore(ArrayList<Task> options, Task target){
	
		 double [] scores = new double[options.size()];
		 for (int i = 0; i < options.size(); i ++){
			 scores[i]= this.getValue(options.get(i).getName(), target.getName());
		 }
		 
		 Task [] tasks = new Task[options.size()];
		 for (int i = 0; i < options.size(); i ++){
			 tasks[i]=options.get(i);
		 }
		 
		 int n = scores.length;
		 
		 double tempScore;
		 Task tempTask;
		 
		 for (int c = 0; c < ( n - 1 ); c++) {
			 for (int d = 0; d < n - c - 1; d++) {
		        if (scores[d] < scores[d+1]) /* For descending order use < */
		        {
		        	
		        	tempScore = scores[d];
		        	tempTask = tasks[d];
		        	
		        	scores[d]   = scores[d+1];
		        	tasks[d]=tasks[d+1];
		          
		        	scores[d+1]=tempScore;
		        	tasks[d+1]=tempTask;
		        	
		        }
		      }
		 }
		 
		 ArrayList<Task> sorted = new ArrayList<Task>();
		 for (int i = 0; i < tasks.length; i++)
			 sorted.add(tasks[i]);
		 
		 return sorted;
	}
	
	public Task samplePositiveSource(ArrayList<Task> options, Task target){
		ArrayList<Task> candidates = new ArrayList<Task>();
		for (int i = 0; i < options.size(); i ++){
			double score_i = this.getValue(options.get(i).getName(), target.getName());
			if (score_i > 0){
				candidates.add(options.get(i));
			}
		}
		
		Random r = new Random();
		return candidates.get(r.nextInt(candidates.size()));
	}
	
	public Task sampleNegativeSource(ArrayList<Task> options, Task target){
		ArrayList<Task> candidates = new ArrayList<Task>();
		for (int i = 0; i < options.size(); i ++){
			double score_i = this.getValue(options.get(i).getName(), target.getName());
			if (score_i < 0){
				candidates.add(options.get(i));
			}
		}
		
		Random r = new Random();
		return candidates.get(r.nextInt(candidates.size()));
	}
	
	public Task getWorstSource(ArrayList<Task> options, Task target){
		int best_index = -1;
		double best_score = 100000.0;
		
		for (int i = 0; i < options.size(); i ++){
			double score_i = this.getValue(options.get(i).getName(), target.getName());
			
			
			
			if (score_i < best_score){
				
				best_index = i;
				best_score = score_i;
			}
		}
		
		return options.get(best_index);
	}
	
	public Task getBestSource(ArrayList<Task> options, Task target){
		int best_index = -1;
		double best_score = -10000.0;
		
		for (int i = 0; i < options.size(); i ++){
			double score_i = this.getValue(options.get(i).getName(), target.getName());
			
			
			
			if (score_i > best_score){
				
				best_index = i;
				best_score = score_i;
			}
		}
		
		return options.get(best_index);
	}
	
	public double getValue(String taskFrom, String taskTo){
		int from = tasks.indexOf(taskFrom);
		int to = tasks.indexOf(taskTo);
		
		return matrix[to][from]; //I know, counter-intuitive
	}
	
	public void printDebug(){
		for (int i = 0; i < matrix.length; i ++){
			for (int j = 0; j < matrix[i].length; j++){
				System.out.print(matrix[i][j]+"\t");
			}
			System.out.println();
		}
	}
	
	public void loadFromFile(String filename){
		ArrayList<double[]> rows = new ArrayList<double[]>();
		
		int n_rows=0; 
		int n_columns;
		
		try {
			BufferedReader BR = new BufferedReader(new FileReader(new File(filename)));
			
			String nextLine;
			while (true){
				nextLine = BR.readLine();
				if (nextLine == null)
					break;
				
				String [] tokens = nextLine.split(",");
				double [] row_i = new double[tokens.length];
				for (int i = 0; i < row_i.length; i ++)
					row_i[i]=Double.parseDouble(tokens[i]);
				
				rows.add(row_i);
				
				n_rows++;
				n_columns = row_i.length;
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	
		
		for (int r = 0; r < rows.size(); r++){
			double [] row_r = rows.get(r);
			for (int c = 0; c < row_r.length; c++)
				matrix[r][c]=row_r[c];
		}
	}
}
