import java.io.*;
import java.util.*;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;


public class Example {

    private static final String FILE_INPUT = "aggregate-results_classified_sample.csv";
    private static final String FILE_OUTPUT = "aggregate-results_classified_sample.arff";

    public static void main(String[] args) throws Exception {

        Instances dataUnfiltered = getInstances(FILE_INPUT,FILE_OUTPUT);
        Instances dataFiltered = filterMeta(dataUnfiltered);
        dataFiltered.setClassIndex(dataFiltered.numAttributes() - 1);

        //System.out.println("Class Attribute : " + data.attribute(data.numAttributes() -1).name());

        // Use a set of classifiers
        Classifier[] models = {
                new RandomForest(),
                //new RandomTree(),
                //new SMO(),
                //new NaiveBayes(),
                //new MultilayerPerceptron()
                //new J48()
        };

        // Run for each model
        for (int j = 0; j < models.length; j++) {
            System.out.println("\n" + models[j].getClass().getSimpleName());
            trainStats(models[j], dataFiltered);
        }

    }


    public static void csvToArff(File csv, File arff) throws IOException {
        // load CSV
        CSVLoader loader = new CSVLoader();
        loader.setSource(csv);
        Instances data = loader.getDataSet();

        // save ARFF
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(arff);
        saver.writeBatch();
    }

    public static Instances getInstances(String csv, String arff) throws Exception {
        File input = getResource(csv);
        File output = getResource(arff);
        csvToArff(input, output);

        BufferedReader datafile = readDataFile(arff);
        Instances data = new Instances(datafile);

        return data;
    }

    public static Instances filterMeta(Instances data) throws Exception {
        //Filter metadatas (here SourceFile, Line, GroupId, ArtifactId, Author, key)

        String [] options = new String [2];
        options[0] = "-R";
        options[1] = "1,2,3,4,5,15";
        Remove remove = new Remove();
        remove.setOptions(options);
        remove.setInputFormat(data);
        data = Filter.useFilter(data, remove);

        return data;
    }


    public static void trainStats(Classifier model, Instances data) throws Exception {
        Evaluation eval = new Evaluation(data);
        eval.crossValidateModel(model, data, 10, new Random(1));

        String recall = Double.toString(eval.recall(0));
        String precision = Double.toString(eval.precision(0));
        String fmeasure = Double.toString(eval.fMeasure(0));
        String accuracy = Double.toString(eval.pctCorrect());
        String confusionMatrix = eval.toMatrixString();

        System.out.println("Estimated Recal : " + recall);
        System.out.println("Estimated Precision : " + precision);
        System.out.println("Estimated F-measure : " + fmeasure);
        System.out.println("Estimated Accuracy : " + accuracy);
        System.out.println("Confusion Matrix : " + confusionMatrix);

    }

    public static String getResourcePath(String fileName) {

        final File f = new File("");
        final String dossierPath = f.getAbsolutePath() + File.separator + fileName;
        return dossierPath;
    }

    public static File getResource(String fileName) {

        final String completeFileName = getResourcePath(fileName);
        File file = new File(completeFileName);
        return file;
    }

    public static BufferedReader readDataFile(String filename) {
        BufferedReader inputReader = null;

        try {
            inputReader = new BufferedReader(new FileReader(filename));
        } catch (FileNotFoundException ex) {
            System.err.println("File not found: " + filename);
        }

        return inputReader;
    }

}