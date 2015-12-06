package Classifier;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Random;

import structures._Corpus;
import structures._Doc;
import structures.annotationType;
import utils.Utils;


public abstract class BaseClassifier {
	protected int m_classNo; //The total number of classes.
	protected int m_featureSize;
	protected _Corpus m_corpus;
	protected ArrayList<_Doc> m_trainSet; //All the documents used as the training set.
	protected ArrayList<_Doc> m_testSet; //All the documents used as the testing set.
	
	protected double[] m_cProbs;
	protected PrintWriter infoWriter;
	
	//for cross-validation
	protected int[][] m_confusionMat, m_TPTable;//confusion matrix over all folds, prediction table in each fold
	protected ArrayList<double[][]> m_precisionsRecalls; //Use this array to represent the precisions and recalls.

	protected String m_debugOutput; // set up debug output (default: no debug output)
	protected BufferedWriter m_debugWriter; // debug output writer
	
	
	public void train() {
		train(m_trainSet);
	}
	
	public abstract void train(Collection<_Doc> trainSet);
	public abstract int predict(_Doc doc);//predict the class label
	public abstract double score(_Doc d, int label);//output the prediction score
	protected abstract void init(); // to be called before training starts
	protected abstract void debug(_Doc d);
	
	public void setInfoWriter(String filePath){
		try{
			infoWriter = new PrintWriter(new File(filePath));
			System.out.println("File Set");
		}
		catch(Exception e){
			e.printStackTrace();
			System.err.println("Info file"+filePath+" Not Found");
		}
	}
	
	public double test() {
		double acc = 0;
		for(_Doc doc: m_testSet){
			doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
			int pred = doc.getPredictLabel(), ans = doc.getYLabel();
			if(ans<0) continue;
			m_TPTable[pred][ans] += 1; //Compare the predicted label and original label, construct the TPTable.
			
			if (pred != ans) {
				if (m_debugOutput!=null && Math.random()<0.2)//try to reduce the output size
					debug(doc);
			} else {//also print out some correctly classified samples
				if (m_debugOutput!=null && Math.random()<0.02)
					debug(doc);
				acc ++;
			}
		}
		
		System.out.print("\nConfusion Matrix\n");
		for(int i=0; i<2;i++){
			for(int j=0; j<2;j++)
				System.out.print(m_TPTable[j][i]+",");
			System.out.println();
		}
		
		m_precisionsRecalls.add(calculatePreRec(m_TPTable));
		return acc /m_testSet.size();
	}
	
	public String getF1String() {
		double[][] PRarray = m_precisionsRecalls.get(m_precisionsRecalls.size()-1);
		StringBuffer buffer = new StringBuffer(128);
		for(int i=0; i<PRarray.length; i++) {
			double p = PRarray[i][0], r = PRarray[i][1];
			//buffer.append(String.format("%d:%.3f ", i, 2*p*r/(p+r)));
			buffer.append(String.format("%.3f,", 2*p*r/(p+r)));
		}
		return buffer.toString().trim();
	}
	
	// Constructor with given corpus.
	public BaseClassifier(_Corpus c) {
		m_classNo = c.getClassSize();
		m_featureSize = c.getFeatureSize();
		m_corpus = c;
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_debugOutput = null;
	}
	
	// Constructor with given dimensions
	public BaseClassifier(int classNo, int featureSize) {
		m_classNo = classNo;
		m_featureSize = featureSize;
		m_corpus = null;
		
		m_trainSet = new ArrayList<_Doc>();
		m_testSet = new ArrayList<_Doc>();
		m_cProbs = new double[m_classNo];
		m_TPTable = new int[m_classNo][m_classNo];
		m_confusionMat = new int[m_classNo][m_classNo];
		m_precisionsRecalls = new ArrayList<double[][]>();
		m_debugOutput = null;
	}
	
	public void setDebugOutput(String filename) {
		if (filename==null || filename.isEmpty())
			return;
		
		File f = new File(filename);
		if(!f.isDirectory()) { 
			if (f.exists()) 
				f.delete();
			m_debugOutput = filename;
		} else {
			System.err.println("Please specify a correct path for debug output!");
		}	
	}
	
	//k-fold Cross Validation.
	public void crossValidation(int k, _Corpus c){
		try {
			if (m_debugOutput!=null){
				m_debugWriter = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(m_debugOutput, false), "UTF-8"));
				m_debugWriter.write(this.toString() + "\n");
			}
			c.shuffle(k);
			int[] masks = c.getMasks();
			ArrayList<_Doc> docs = c.getCollection();
			//Use this loop to iterate all the ten folders, set the train set and test set.
			for (int i = 0; i < k; i++) {
				for (int j = 0; j < masks.length; j++) {
					//more for testing
//					if( masks[j]==(i+1)%k || masks[j]==(i+2)%k ) // || masks[j]==(i+3)%k 
//						m_trainSet.add(docs.get(j));
//					else
//						m_testSet.add(docs.get(j));
					
					//more for training
					if(masks[j]==i) 
						m_testSet.add(docs.get(j));
					else
						m_trainSet.add(docs.get(j));
				}
				
				long start = System.currentTimeMillis();
				train();
				double accuracy = test();
				
				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
				//infoWriter.write(getF1String());
				m_trainSet.clear();
				m_testSet.clear();
			}
			calculateMeanVariance(m_precisionsRecalls);	
		
			if (m_debugOutput!=null)
				m_debugWriter.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//k-fold Cross Validation.
	public void crossValidationNahid(int k, boolean m_randomFold, boolean m_LoadnewEggInTrain) {
			m_trainSet = new ArrayList<_Doc>();
			m_testSet = new ArrayList<_Doc>();
			
			HashMap<Integer, double[]> tpt = new HashMap<Integer, double[]>();
			
			double[] perf;
			int amazonTrainsetRatingCount[] = {0,0,0,0,0};
			int amazonRatingCount[] = {0,0,0,0,0};
			
			int newEggRatingCount[] = {0,0,0,0,0};
			int newEggTrainsetRatingCount[] = {0,0,0,0,0};
			
			
			/*if(m_randomFold==true){
				perf = new double[k];
				//m_corpus.shuffle(k);
				int[] masks = m_corpus.getMasks();
				ArrayList<_Doc> docs = m_corpus.getCollection();
				//Use this loop to iterate all the ten folders, set the train set and test set.
				for (int i = 0; i < k; i++) {
					m_corpus.shuffle(k);
					for (int j = 0; j < masks.length; j++) {
						if( masks[j]==i ){ 
							m_testSet.add(docs.get(j));
						}
						else{ 
							m_trainSet.add(docs.get(j));
						}
						

						if(m_trainSet.size()<=0.8*docs.size())
							m_trainSet.add(docs.get(j));
						else
							m_testSet.add(docs.get(j));
					}
					
					System.out.println("Fold number "+i);
					System.out.println("Train Set Size "+m_trainSet.size());
					System.out.println("Test Set Size "+m_testSet.size());

					long start = System.currentTimeMillis();
					train();
					double accuracy = test();
					
					System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
					m_trainSet.clear();
					m_testSet.clear();
				}
			}else */if(m_randomFold==true){
				
				perf = new double[k];
				ArrayList<_Doc> neweggDocs= new ArrayList<_Doc>() ;
				
				for(_Doc d:m_corpus.getCollection()){
					if(d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED)
						neweggDocs.add(d);
				}
				
				// shuffling only newEgg docs
				int[] masks = new int [neweggDocs.size()]; 
				Random rand = new Random(0);
				for(int i=0; i< masks.length; i++) {
					masks[i] = rand.nextInt(k);
				}
				
				//Use this loop to iterate all the k folders, set the train set and test set.
				for (int i = 0; i < k; i++) {
					
					// adding one fold of train and test from newEgg
					for (int j = 0; j < masks.length; j++) {
						if( masks[j]==i ){ 
							m_testSet.add(neweggDocs.get(j));
						}
						else{ 
							m_trainSet.add(neweggDocs.get(j));
						}
					}
					
					//adding all the data from amazon in trainset
					int index = 0;
					
					for(int a=0; a<=25000;a=a+5000){
						System.out.println("a:"+ a);
						
						
						// actiavte this for EM-Naive Bayes
						/*if(a!=0){
							int m = 0;
							int l = index;
							for(; ;l++){
								_Doc d = m_corpus.getCollection().get(l);
								if(m>5000)
									break;
								if(d.getAnnotationType()==annotationType.UNANNOTATED){
									m_trainSet.add(d);
									m++;
								}
								
							}
							index = l;

						}*/
						
						System.out.println("Fold number "+i);
						System.out.println("Train Set Size "+m_trainSet.size());
						System.out.println("Test Set Size "+m_testSet.size());

						long start = System.currentTimeMillis();
						train();
						//double accuracy = test();
						
						int precision_recall [][] = {{0,0},{0,0}};
						for(_Doc doc: m_testSet){
							doc.setPredictLabel(predict(doc)); //Set the predict label according to the probability of different classes.
							int pred = doc.getPredictLabel(), ans = doc.getYLabel();
							if(pred == -1 || ans ==-1) continue;
							precision_recall[ans][pred] += 1; //Compare the predicted label and original label, construct the TPTable.
				
						}
					    
						System.out.println("\nConfusion Matrix");
						//infoWriter.println("Confusion Matrix");
						for(int z=0; z<2; z++)
						{
							for(int j=0; j<2; j++)
							{
								System.out.print(precision_recall[z][j]+",");
							//	infoWriter.print(precision_recall[l][j]+",");
							}
							System.out.println();
							//infoWriter.println();
						}
						
					    double pros_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
						double cons_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);
						
						double pros_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
						double cons_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);
						
						double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
						double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
						
						double result [] = {pros_f1,cons_f1};
						
						tpt.put(i+a, result);
					    
						System.out.format("%s Train/Test finished in %.2f seconds with pros F1 %.4f and cons F1 %.4f ..\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, pros_f1, cons_f1);
						
						
					}// amazon trainSize loop ends
			
					m_trainSet.clear();
					m_testSet.clear();
				}
			}
			
			else {
				k = 1;
				perf = new double[k];
			    int totalNewqEggDoc = 0;
			    int totalAmazonDoc = 0;
				for(_Doc d:m_corpus.getCollection()){
					if(d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED){
						newEggRatingCount[d.getYLabel()]++;
						totalNewqEggDoc++;
						}
					else if(d.getAnnotationType()==annotationType.UNANNOTATED){
						amazonRatingCount[d.getYLabel()]++;
						totalAmazonDoc++;
					}
				}
				System.out.println("Total New Egg Doc:"+totalNewqEggDoc);
				System.out.println("Total Amazon Doc:"+ totalAmazonDoc);
				
				int amazonTrainSize = 0;
				int amazonTestSize = 0;
				int newEggTrainSize = 0;
				int newEggTestSize = 0;
				
				for(_Doc d:m_corpus.getCollection()){
					
					if(d.getAnnotationType()==annotationType.UNANNOTATED){ // from Amazon
						int rating = d.getYLabel();
						
						if(amazonTrainsetRatingCount[rating]<=0.8*amazonRatingCount[rating]){
							m_trainSet.add(d);
							amazonTrainsetRatingCount[rating]++;
							amazonTrainSize++;
						}else{
							m_testSet.add(d);
							amazonTestSize++;
						}
					}
					
					if(m_LoadnewEggInTrain==true && (d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED)) {
						
						int rating = d.getYLabel();
						if(newEggTrainsetRatingCount[rating]<=0.8*newEggRatingCount[rating]){
							m_trainSet.add(d);
							newEggTrainsetRatingCount[rating]++;
							newEggTrainSize++;
						}else{
							m_testSet.add(d);
							newEggTestSize++;
						}
						
					}
					if(m_LoadnewEggInTrain==false && (d.getAnnotationType()==annotationType.ANNOTATED || d.getAnnotationType()==annotationType.PARTIALLY_ANNOTATED)) {
						int rating = d.getYLabel();
						if(newEggTrainsetRatingCount[rating]<=0.8*newEggRatingCount[rating]){
							// Do nothing simply ignore it make for different set
							//m_trainSet.add(d);
							newEggTrainsetRatingCount[rating]++;
							//newEggTrainSize++;
						}else{
							m_testSet.add(d);
							newEggTestSize++;
						}
					}
				}
				
				System.out.println("Neweeg Train Size: "+newEggTrainSize+" test Size: "+newEggTestSize);
				
				System.out.println("Amazon Train Size: "+amazonTrainSize+" test Size: "+amazonTestSize);
				
				for(int i=0; i<amazonTrainsetRatingCount.length; i++){
					System.out.println("Rating ["+i+"] and Amazon TrainSize:"+amazonTrainsetRatingCount[i]+" and newEgg TrainSize:"+newEggTrainsetRatingCount[i]);
				}
		
				System.out.println("Combined Train Set Size "+m_trainSet.size());
				System.out.println("Combined Test Set Size "+m_testSet.size());
				
				long start = System.currentTimeMillis();
				train();
				double accuracy = test();
				
				System.out.format("%s Train/Test finished in %.2f seconds with accuracy %.4f and F1 (%s)...\n", this.toString(), (System.currentTimeMillis()-start)/1000.0, accuracy, getF1String());
						
			}
			//output the performance statistics
			calculateMeanVariance(m_precisionsRecalls);	
			/*
			// calculate statistics for new folding mechanism
			double trainSizeSum [][] = {{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};
			for(int size = 0; size<=5; size = size+1){
				for(int fold = 0; fold<k;fold++){
					double tmp [] = tpt.get(fold+size*5000);
					trainSizeSum[size][0] += tmp[0];
					trainSizeSum[size][1] += tmp[1];
				}
			}
			
			for(int size = 0; size<=5; size = size+1){
				System.out.println("Train size:"+size*1000 +", Pros F1:"+trainSizeSum[size][0]/k+", Cons F1:"+trainSizeSum[size][1]/k);
			}*/
			
		}	
	
	abstract public void saveModel(String modelLocation);
	
	//Calculate the precision and recall for one folder tests.
	public double[][] calculatePreRec(int[][] tpTable) {
		double[][] PreRecOfOneFold = new double[m_classNo][2];
		
		for (int i = 0; i < m_classNo; i++) {
			PreRecOfOneFold[i][0] = (double) tpTable[i][i] / (Utils.sumOfRow(tpTable, i) + 0.001);// Precision of the class.
			PreRecOfOneFold[i][1] = (double) tpTable[i][i] / (Utils.sumOfColumn(tpTable, i) + 0.001);// Recall of the class.
		}
		
		for (int i = 0; i < m_classNo; i++) {			
			for(int j=0; j< m_classNo; j++) {
				m_confusionMat[i][j] += tpTable[i][j];
				tpTable[i][j] = 0; // clear the result in each fold
			}
		}
		return PreRecOfOneFold;
	}
	
	public void printConfusionMat() {
		for(int i=0; i<m_classNo; i++){
			System.out.format("\t%d", i);
			infoWriter.format("\t%d", i);
		}
		
		double total = 0, correct = 0;
		double[] columnSum = new double[m_classNo], prec = new double[m_classNo];
		System.out.println("\tP");
		infoWriter.println("\tP");
		for(int i=0; i<m_classNo; i++){
			System.out.format("%d", i);
			infoWriter.format("%d", i);
			double sum = 0; // row sum
			for(int j=0; j<m_classNo; j++) {
				System.out.format("\t%d", m_confusionMat[i][j]);
				infoWriter.format("\t%d", m_confusionMat[i][j]);
				sum += m_confusionMat[i][j];
				columnSum[j] += m_confusionMat[i][j];
				total += m_confusionMat[i][j];
			}
			correct += m_confusionMat[i][i];
			prec[i] = m_confusionMat[i][i]/sum;
			System.out.format("\t%.4f\n", prec[i]);
			infoWriter.format("\t%.4f\n", prec[i]);
		}
		
		System.out.print("R");
		infoWriter.print("R");
		for(int i=0; i<m_classNo; i++){
			columnSum[i] = m_confusionMat[i][i]/columnSum[i]; // recall
			System.out.format("\t%.4f", columnSum[i]);
			infoWriter.format("\t%.4f", columnSum[i]);
		}
		System.out.format("\t%.4f", correct/total);
		infoWriter.format("\t%.4f", correct/total);
		
		System.out.print("\nF1");
		infoWriter.print("\ncompleteF1");
		for(int i=0; i<m_classNo; i++){
			System.out.format(",%.4f", 2.0 * columnSum[i] * prec[i] / (columnSum[i] + prec[i]));
			infoWriter.format(",%.4f", 2.0 * columnSum[i] * prec[i] / (columnSum[i] + prec[i]));
		}
		System.out.println();
		infoWriter.println();
		
	}
	
	//Calculate the mean and variance of precision and recall.
	public double[][] calculateMeanVariance(ArrayList<double[][]> prs){
		//Use the two-dimension array to represent the final result.
		double[][] metrix = new double[m_classNo][4]; 
			
		double precisionSum = 0.0;
		double precisionVarSum = 0.0;
		double recallSum = 0.0;
		double recallVarSum = 0.0;

		//i represents the class label, calculate the mean and variance of different classes.
		for(int i = 0; i < m_classNo; i++){
			precisionSum = 0;
			recallSum = 0;
			// Calculate the sum of precisions and recalls.
			for (int j = 0; j < prs.size(); j++) {
				precisionSum += prs.get(j)[i][0];
				recallSum += prs.get(j)[i][1];
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][0] = precisionSum/prs.size();
			metrix[i][1] = recallSum/prs.size();
		}

		// Calculate the sum of variances of precisions and recalls.
		for (int i = 0; i < m_classNo; i++) {
			precisionVarSum = 0.0;
			recallVarSum = 0.0;
			// Calculate the sum of precision variance and recall variance.
			for (int j = 0; j < prs.size(); j++) {
				precisionVarSum += (prs.get(j)[i][0] - metrix[i][0])*(prs.get(j)[i][0] - metrix[i][0]);
				recallVarSum += (prs.get(j)[i][1] - metrix[i][1])*(prs.get(j)[i][1] - metrix[i][1]);
			}
			
			// Calculate the means of precisions and recalls.
			metrix[i][2] = Math.sqrt(precisionVarSum/prs.size());
			metrix[i][3] = Math.sqrt(recallVarSum/prs.size());
		}
		
		// The final output of the computation.
		System.out.println("*************************************************");
		System.out.format("The final result of %s is as follows:\n", this.toString());
		infoWriter.format("The final result of %s is as follows:\n", this.toString());
		System.out.println("The total number of classes is " + m_classNo);
		infoWriter.println("The total number of classes is " + m_classNo);
		
		for(int i = 0; i < m_classNo; i++){
			System.out.format("Class %d:\tprecision(%.3f+/-%.3f)\trecall(%.3f+/-%.3f)\n", i, metrix[i][0], metrix[i][2], metrix[i][1], metrix[i][3]);
			infoWriter.format("Class %d:\tprecision(%.3f+/-%.3f)\trecall(%.3f+/-%.3f)\n", i, metrix[i][0], metrix[i][2], metrix[i][1], metrix[i][3]);
			
		}
		printConfusionMat();
		infoWriter.flush();
		infoWriter.close();
		return metrix;
	}
}
