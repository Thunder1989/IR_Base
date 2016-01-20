package mains;

import java.io.*;
import java.util.*;
import java.text.ParseException;

import Analyzer.AspectAnalyzer;
import Classifier.metricLearning.L2RMetricLearning;
import Classifier.metricLearning.LinearSVMMetricLearning;
import Classifier.semisupervised.GaussianFieldsByRandomWalk;
import Classifier.supervised.SVM;
import structures._Corpus;

public class CRFMain {
    public static void ReadFile(String fileName) {
        BufferedReader file = null;
        try {
                file = new BufferedReader(new FileReader(fileName));
                String line;
                while ((line = file.readLine()) != null) {
                    System.out.println(line);
                }
                file.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
    }
    
    public static void main(String[] args) throws IOException, ParseException {
        System.out.println("Test");
        String inputFile = "./test.csv";
        ReadFile(inputFile);
    }
}
