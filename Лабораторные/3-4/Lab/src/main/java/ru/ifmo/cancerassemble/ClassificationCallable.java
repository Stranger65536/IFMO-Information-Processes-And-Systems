package ru.ifmo.cancerassemble;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.evaluation.ThresholdCurve;
import weka.core.Instances;

import java.util.Random;
import java.util.concurrent.Callable;

class ClassificationCallable implements Callable<ClassificationResult> {
    private static final int CLASS_INDEX = 0;
    private static final Random RANDOM = new Random();

    private final Classifier classifier;
    private final Instances data;
    private final Evaluation evaluation;

    ClassificationCallable(final Classifier classifier, final Instances data) {
        this.classifier = classifier;
        this.data = data;

        try {
            this.evaluation = new Evaluation(data);
        } catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
    }

    @Override
    public ClassificationResult call() {
        try {
            final long before = System.currentTimeMillis();
            classifier.buildClassifier(data);
            evaluation.crossValidateModel(classifier, data, 10, RANDOM);
            final long after = System.currentTimeMillis();

            final ThresholdCurve thresholdCurve = new ThresholdCurve();
            final Instances curve = thresholdCurve.getCurve(evaluation.predictions(), CLASS_INDEX);

            final double[] tp = curve.attributeToDoubleArray(curve.attribute("True Positives").index());
            final double tpc = evaluation.numTruePositives(CLASS_INDEX);
            final double fnc = evaluation.numFalseNegatives(CLASS_INDEX);
            final double[] fp = curve.attributeToDoubleArray(curve.attribute("False Positives").index());
            final double fpc = evaluation.numFalsePositives(CLASS_INDEX);
            final double tnc = evaluation.numTrueNegatives(CLASS_INDEX);
            final double auc = evaluation.areaUnderROC(CLASS_INDEX);
            final double tnr = evaluation.trueNegativeRate(CLASS_INDEX);
            final double tpr = evaluation.truePositiveRate(CLASS_INDEX);

            return new ClassificationResult(tp, tpc, fp, fpc, fnc, tnc, tnr, tpr, auc, after - before, classifier);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }
}
