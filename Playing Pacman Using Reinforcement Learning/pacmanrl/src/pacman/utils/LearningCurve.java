package pacman.utils;



/**
 * Represents a curve showing agent performance against episodes trained.
 */
public class LearningCurve {

	public final int length; // Total points
	public final int width; // Episodes per point
	private double[] score; // Agent performance
	private double[][] data; // Extra info
	private double[] steps;		// Number of steps taken to complete game

	/** Start with an empty curve. */
	public LearningCurve(int length, int width) {
		this.length = length;
		this.width = width;
		score = new double[length];
		data = new double[length][];
		steps = new double[length];
	}
	
	/** Load from a file. */
	public LearningCurve(int length, int width, String filename) {
		this(length, width);
		
		DataFile file = new DataFile(filename);
		for (int x=0; x<length; x++) {
			String[] line = file.nextLine().split("\t");
			score[x] = Double.parseDouble(line[1]);
			if (line.length < 3)
				continue;
			steps[x] = Double.parseDouble(line[2]);
			data[x] = new double[line.length-3];
			for (int d=0; d<data[x].length; d++)
				data[x][d] = Double.parseDouble(line[d+2]);
		}
		file.close();
	}
	
	/** Average together a set of curves with the same dimensions. */
	public LearningCurve(LearningCurve[] curves) {
		this(curves[0].length, curves[0].width);
		
		for (int x=0; x<length; x++) {
			data[x] = new double[curves[0].data[x].length];
			                     
			for (LearningCurve curve : curves) {
				score[x] += curve.score[x];
				steps[x] += curve.steps[x];
				for (int d=0; d<data[x].length; d++)
					data[x][d] += curve.data[x][d];
			}
			score[x] /= curves.length;
			steps[x] /= curves.length;
			for (int d=0; d<data[x].length; d++)
				data[x][d] /= curves.length;
		}
	}
	
	/** Compute a time series of the standard deviation or variance given a set of curves
	 * only works on the score */
	public static LearningCurve getScoreVarianceCurve(LearningCurve[] curves, boolean useStdev){
		LearningCurve LC = new LearningCurve(curves[0].length, curves[0].width);
		
		for (int x=0; x<LC.length; x++) {
			//LC.data[x] = new double[curves[0].data[x].length];
			                    
			double mean_score_x = 0;
			double [] scores_at_x = new double[curves.length];
			int t = 0;
			for (LearningCurve curve : curves) {
				mean_score_x += curve.score[x];
				scores_at_x[t] = curve.score[x]; t++;
			}
			
			mean_score_x /= curves.length;
			
			double v_at_x = Stats.variance(scores_at_x, mean_score_x);
			
			//if we're using standard deviation instead
			if (useStdev)
				v_at_x = Math.sqrt(v_at_x);
			
			LC.set(x, v_at_x, new double[curves[0].data[x].length],0);
		}
		
		return LC;
	}

	/** Assign a point. */
	public void set(int x, double score, double[] data, double steps) {
		this.score[x] = score;
		this.data[x] = data;
		this.steps[x] = steps;
	}

	/** Compute area under the score curve, starting from start */
	public double area(int start) {
		double area = 0;
		for (int x=start; x<length; x++)
			area += score[x];
		return area;
	}
	public double area(){
		return area(0);
	}
	
	/** Save to a file. */
	public void save(String filename) {
		DataFile file = new DataFile(filename);
		file.clear();
		for (int x=0; x<length; x++) {
			file.append((x*width)+"\t"+score[x]);
			file.append("\t"+steps[x]);
			try {
			for (int d=0; d<data[x].length; d++)
				file.append("\t"+data[x][d]);
			} catch(Exception e){}
			file.append("\n");
		}
		file.close();
	}
	
	public double [] getScoreArray(){
		return score;
	}
	
	public double getScore(int index){
		if (index < length){
			return score[index];
		}
		return -1;
	}
}
