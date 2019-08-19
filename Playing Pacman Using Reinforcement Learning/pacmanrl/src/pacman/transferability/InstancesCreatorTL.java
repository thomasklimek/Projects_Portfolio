package pacman.transferability;

import java.util.ArrayList;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;

public class InstancesCreatorTL {

	TransferabilityMatrix M;
	
	ArrayList<String> attribute_names;
	ArrayList<String> attribute_types;
	
	double [] attr_min_values;
	double [] attr_max_values;
	double [] ranges;
	
	
	public Instances header;
	
	String featureType;
	
	public InstancesCreatorTL(String f){
		featureType = f;
	}
	
	public void setMinMaxRanges(double [] min, double [] max){
		attr_min_values = min;
		attr_max_values = max;
		ranges = new double[min.length];
		for (int i = 0; i < ranges.length; i++){
			ranges[i]=attr_max_values[i]-attr_min_values[i];
		}
	}
	
	public void setTransferabilityMatrix(TransferabilityMatrix M){
		this.M = M;
	}
	
	public void setAttributeInfo(ArrayList<String> attribute_names,
								ArrayList<String> attribute_types){
		this.attribute_names=attribute_names;
		this.attribute_types=attribute_types;
	}
	
	public Instances createInstances(ArrayList<Task[]> task_pairs){
		Instances data = new Instances(header);
		
		for (int i = 0; i < task_pairs.size(); i++){
			Task [] pair_i = task_pairs.get(i);
			Instance x_i = this.createInstance(pair_i[0],pair_i[1]);
			x_i.setDataset(data);
			data.add(x_i);
		}
		
		
		return data;
	}
	
	public Instance createInstance(Task task_from, Task task_to){
		Instance x = new DenseInstance(header.numAttributes());
		x.setDataset(header);
		
		ArrayList<Double> values_from = task_from.getAttributeValues();
		ArrayList<Double> values_to = task_to.getAttributeValues();
		
		for (int i = 0; i < attribute_names.size(); i ++){
			if (attribute_types.get(i).equals("numeric")){
				
				double v_i = 0;
				
				
				if (featureType == "ratio") //ratio
					v_i = values_from.get(i).doubleValue() / values_to.get(i).doubleValue() ;	
				else if (featureType == "difference")
					v_i = values_from.get(i).doubleValue() - values_to.get(i).doubleValue() ;	
				else if (featureType == "pct_difference")
					v_i = (values_to.get(i).doubleValue() - values_from.get(i).doubleValue()) /  values_from.get(i).doubleValue();
				else if (featureType == "scaled_difference"){
					
					v_i = (values_from.get(i).doubleValue() - values_to.get(i).doubleValue())/ranges[i];
					
				}
					
				if (Double.isNaN(v_i)){ //division by 0 
					x.setMissing(i);
				}
				else if (Double.isInfinite(v_i)){
					x.setMissing(i);
				}
				else
					x.setValue(i, v_i);
			}
			else {
				int a = (int)values_from.get(i).doubleValue();
				int b = (int)values_to.get(i).doubleValue();

				x.setValue(i, new String(a+"_to_"+b));
			}
		}
		
		x.setClassValue(M.getValue(task_from.name, task_to.name));
		
		return x;
	}
	
	public void generateHeader(){
		FastVector attrInfo = new FastVector();
		
		for (int i = 0; i < attribute_names.size(); i ++){
			
			if (attribute_types.get(i) == "numeric"){
				Attribute a_i = new Attribute(new String("attr_"+attribute_names.get(i)));
				attrInfo.add(a_i);
			}
			else {
				FastVector attr_i_values = new FastVector();
				
				if (attribute_names.get(i).equals("ghost_type")){
					//attr_i_values.add("0");
					//attr_i_values.add("1");
					//attr_i_values.add("2");
					for (int gtype = 0; gtype < 3; gtype++){
						for (int gtype_to = 0; gtype_to < 3; gtype_to++){
							attr_i_values.add(gtype+"_to_"+gtype_to);
						}
					}
				}
				
				Attribute a_i = new Attribute("attr_"+attribute_names.get(i),attr_i_values);
				attrInfo.add(a_i);
			}
			
		}
		
		
		//class atrribute is numeric
		Attribute class_attr = new Attribute("class");

		attrInfo.add(class_attr);
		
		
		
		header = new Instances("data",attrInfo,0);
		header.setClassIndex(header.numAttributes()-1);
	}
	
}
