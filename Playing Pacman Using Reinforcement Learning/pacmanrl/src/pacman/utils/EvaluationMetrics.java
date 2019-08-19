package pacman.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import pacman.utils.LearningCurve;

public class EvaluationMetrics {

	
	public static double getAvgReward(double [] series, int start, int end){
		double rd = 0;
		int c = 0;
		
		for (int i = start; i < end; i ++){
			rd+=series[i];
			c++;
		}
		
		return rd / (double)c;
	}
	
	/**
	 * Computes the average reward for the first k episodes for each condition and 
	 * then returns the difference
	 * 	
	 * positive result means there is jumpstart benefit from transfer, negative means there 
	 * is detriment from transfer
	 * 
	 * when k = 1, this is equiavelent to the standard jumpstart metric describe in Matt's thesis
	 * 
	 */
	public static double jumpStartK(double [] reward_default, double [] reward_transfer, int k){
		double rd = 0;
		double rt = 0;
		
		for (int i = 0; i < k; i ++){
			rd+=reward_default[i];
			rt+=reward_transfer[i];
		}
		
		rd = rd / (double)k;
		rt = rt / (double)k;
		
		return rt-rd;
		
	}
	
	/**
	 * Computes the average reward for the first k episodes for each condition and 
	 * then returns the ratio
	 * 
	 */
	public static double jumpStartRatioK(double [] reward_default, double [] reward_transfer, int k){
		double rd = 0;
		double rt = 0;
		
		for (int i = 0; i < k; i ++){
			rd+=reward_default[i];
			rt+=reward_transfer[i];
		}
		
		rd = rd / (double)k;
		rt = rt / (double)k;
		
		if (rd == 0)
			rd = Double.MIN_VALUE;
		
		double ratio = rt/rd;
		
		
		
		return ratio;
	}
	
	
	/** Computes the area between the curves. Note that the x axis is measured in episodes, not steps! */
	public static double transferDifference(LearningCurve transferCurve, LearningCurve independentCurve, int independentStart){		
		if (transferCurve.length != independentCurve.length - independentStart){
			System.err.println("Mismatching curves for transfer difference!");
		}
		return (transferCurve.area() - independentCurve.area(independentStart));
	}
	
	/** Computes the increase in reward relative to using no transfer */
	public static double transferRatio(LearningCurve transferCurve, LearningCurve independentCurve){		
		return (transferCurve.area() - independentCurve.area()) / (independentCurve.area());
	}
	
	public static double getExpectedRewardInRange(double [][] curve_erg, int [] range){
		double R = 0;
		int k = 0;
		
		for (int i = 0; i < curve_erg.length; i++){
			if (curve_erg[i][2] >= (double)range[0] && curve_erg[i][2] <= (double)range[1]){
				R += curve_erg[i][1];
				k++;
			}
		}
		
		return R/(double)k;
	}
	
	/*
	 * The input should be a double array of size n x 3 where [i][0] is episode
	 * [i][1] is reward and [i][2] is gamestep
	 * 
	 */
	public static double getAreaUnderCurve(double [][] curve_erg, int game_step_threshold){
		double sum = 0;
		
		for (int i = 0; i < curve_erg.length; i++){
			if (curve_erg[i][2] <= (double)game_step_threshold || game_step_threshold == -1){
				sum += curve_erg[i][1];
			}
		}
		
		return sum;
	}
	
	public static double [][] loadCurveXD(String filename){
		double [][] R;
		
		ArrayList<double[]> entries = new ArrayList<double[]>();
		int dim = -1;
		
		try {
			BufferedReader BR = new BufferedReader(new FileReader(new File(filename)));
			
			while (true){
				String line = BR.readLine();
				if (line == null)
					break;
				
				String [] tokens = line.split("\t");
				dim = tokens.length;
				
				double [] x = new double[dim];
				for (int i = 0; i < x.length; i++){
					x[i] = Double.parseDouble(tokens[i]);
				}
				
				entries.add(x);
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		R = new double[entries.size()][dim];
		for (int i = 0; i < entries.size(); i++){
			R[i]=entries.get(i);
		}
		
		return R;
	}
}
