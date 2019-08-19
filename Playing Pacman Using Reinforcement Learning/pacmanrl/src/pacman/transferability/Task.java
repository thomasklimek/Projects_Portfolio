package pacman.transferability;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class Task {

	String name;
	
	ArrayList<String> attribute_names;
	ArrayList<Double> values;
	ArrayList<String> attribute_types; //either "numeric" or "nominal"
	
	public Task(String name){
		this.name = name;
		
		attribute_names = new ArrayList<String>();
		attribute_types = new ArrayList<String>();
		values = new ArrayList<Double>();
	}
	
	public String getName(){
		return name;
	}
	
	public ArrayList<String> getAttributeNames(){
		return attribute_names;
	}
	
	public ArrayList<String> getAttributeTypes(){
		return attribute_types;
	}
	
	public ArrayList<Double> getAttributeValues(){
		return values;
	}
	
	public void printDebug(){
		System.out.println("Task name:\t"+name);
		for (int i = 0; i < attribute_names.size(); i++){
			System.out.println("\t"+attribute_names.get(i)+"\t"+values.get(i).doubleValue()+"\t"+attribute_types.get(i));
		}
	}
	
	public void addAttributeValue(String attribute, Double value, String type){
		attribute_names.add(attribute);
		attribute_types.add(type);
		values.add(value);
	}
	
	public void addNumericalAttributesFromFile(String filename){
		try {
			BufferedReader BR = new BufferedReader(new FileReader(new File(filename)));
			
			while(true){
				String line = BR.readLine();
				
				if (line == null)
					break;
				
				String [] tokens = line.split("\t");
				if (tokens.length == 2){
					attribute_names.add(tokens[0]);
					attribute_types.add("numeric");
					values.add(new Double(Double.parseDouble(tokens[1])));
				}
			}
			
			BR.close();
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
