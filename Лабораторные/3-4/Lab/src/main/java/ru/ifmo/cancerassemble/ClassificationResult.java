package ru.ifmo.cancerassemble;

import lombok.Value;
import weka.classifiers.Classifier;

@Value
class ClassificationResult {
    double[] truePositives;
    double truePositivesCount;
    double[] falsePositives;
    double falsePositivesCount;
    double falseNegativesCount;
    double trueNegativesCount;
    double trueNegativesRate;
    double truePositivesRate;
    double areaUnderCurve;
    long executionTime;
    Classifier classifier;

    double getMatthewsCorrelationCoefficient() {
        return (truePositivesCount * trueNegativesCount -
                falsePositivesCount * falseNegativesCount) /
                Math.sqrt((truePositivesCount + falsePositivesCount) *
                        (truePositivesCount + falseNegativesCount) *
                        (trueNegativesCount + falsePositivesCount) *
                        (trueNegativesCount + falseNegativesCount));
    }
}

