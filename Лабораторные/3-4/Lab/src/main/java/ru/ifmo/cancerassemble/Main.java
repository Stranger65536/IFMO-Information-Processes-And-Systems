package ru.ifmo.cancerassemble;

import com.google.common.base.Charsets;
import com.google.common.primitives.Doubles;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.data.xy.DefaultXYDataset;
import org.jfree.data.xy.XYDataset;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.BayesianLogisticRegression;
import weka.classifiers.bayes.DMNBtext;
import weka.classifiers.functions.RBFNetwork;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.VotedPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.lazy.LWL;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.ConjunctiveRule;
import weka.classifiers.rules.JRip;
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.REPTree;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import javax.swing.*;
import javax.swing.table.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.*;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.*;
import java.util.stream.IntStream;

public class Main {
    private static final String INPUT_FILE_PATH = "data.arff";

    private static final String MAIN_FRAME_TITLE = "ROC Curves";
    private static final String INITIAL_PROGRESS_LABEL = "In a queue";
    private static final String PROGRESS_FRAME_TITLE = "Progress Frame";
    private static final String VERTICAL_AXIS_LABEL = "True Positive Rate";
    private static final String HORIZONTAL_AXIS_LABEL = "False Positive Rate";

    private static final float ROC_LOWER_BOUND = 0.0f;
    private static final float ROC_UPPER_BOUND = 1.0f;

    private static final float STROKE_WIDTH = 2.5f;
    private static final float DASH_GROW_POW = 0.75f;
    private static final float STROKE_MITERLIMIT = 5.0f;
    private static final float STROKE_DASH_PHASE = 0.0f;

    private static final int HEADER_ROW = 0;
    private static final int FIRST_COLUMN = 0;
    private static final boolean HAS_NO_FOCUS = false;
    private static final boolean DOES_NOT_SELECTED = false;

    private static final boolean NO_URLS = false;
    private static final boolean ENABLE_LEGEND = true;
    private static final boolean ENABLE_TOOLTIPS = true;

    private static final int DEFAULT_WINDOW_WIDTH = 1280;
    private static final int DEFAULT_WINDOW_HEIGHT = 1024;
    private static final Dimension SCREEN_SIZE = Toolkit.getDefaultToolkit().getScreenSize();

    private static final float GRID_STEP = 0.025f;
    private static final int TABLE_COLUMN_MARGIN = 10;
    private static final Paint GRID_COLOR = Color.DARK_GRAY;
    private static final Paint BACKGROUND_COLOR = new Color(235, 235, 235);

    private static final String[] TABLE_HEADER_LABELS = {
            "Classifier", "AUC", "Sensitivity", "Specification", "MCC", "Compute time"};

    private static DefaultTableModel getReadOnlyTableModel(
            final String[] headerLabels,
            final int rows) {
        return new DefaultTableModel(headerLabels, rows) {
            @Override
            public boolean isCellEditable(final int row, final int column) {
                return false;
            }
        };
    }

    public static void main(final String[] args) {
        final List<Classifier> classifiers = initializeClassifiers();
        final Instances data = readData(INPUT_FILE_PATH);
        final JTable table = prepareProgressTable(classifiers);
        final Map<Integer, Boolean> visibilityStatuses = new ConcurrentHashMap<>(classifiers.size());
        final Map<Integer, Integer> donePositions = new ConcurrentHashMap<>(classifiers.size());
        final ChartComponents chartComponents = configureAndShowMainWindow();

        configureAndShowProgressFrame(classifiers, table, visibilityStatuses,
                donePositions, chartComponents.getChart());
        performClassification(data, classifiers, chartComponents, table, donePositions);
    }

    private static JTable prepareProgressTable(final Collection<Classifier> classifiers) {
        final JTable table = new JTable(getReadOnlyTableModel(TABLE_HEADER_LABELS, classifiers.size()));
        table.getTableHeader().setReorderingAllowed(false);
        return table;
    }

    private static List<Classifier> initializeClassifiers() {
        final List<Classifier> classifiers = new CopyOnWriteArrayList<>();

        classifiers.add(new AdaBoostM1());
        classifiers.add(new BayesianLogisticRegression());
        classifiers.add(new ConjunctiveRule());
        classifiers.add(new DMNBtext());
        classifiers.add(new DecisionStump());
        classifiers.add(new JRip());
        classifiers.add(new LWL());
        classifiers.add(new RBFNetwork());
        classifiers.add(new REPTree());
        classifiers.add(new RandomForest());
        classifiers.add(new SimpleLogistic());
        classifiers.add(new IBk());
        classifiers.add(new VotedPerceptron());

        Collections.shuffle(classifiers);

        return classifiers;
    }

    private static Instances readData(final String filePath) {
        try (final BufferedReader reader = new BufferedReader(
                new InputStreamReader(new FileInputStream(filePath), Charsets.UTF_8))) {
            final Instances data = new Instances(reader);
            data.setClass(data.attribute(data.numAttributes() - 1));
            return data;
        } catch (final FileNotFoundException e) {
            throw new IllegalArgumentException("Can't find file " + filePath, e);
        } catch (final IOException e) {
            throw new IllegalArgumentException("Error parsing file " + filePath, e);
        }
    }

    private static ChartComponents configureAndShowMainWindow() {
        final DefaultXYDataset dataset = new DefaultXYDataset();
        final JFreeChart chart = prepareLineChart(dataset);

        preparePlot((XYPlot) chart.getPlot(), BACKGROUND_COLOR, GRID_COLOR, GRID_STEP);

        final ChartPanel panel = new ChartPanel(chart);
        final JFrame jf = prepareMainFrame(panel, MAIN_FRAME_TITLE,
                DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);

        jf.setVisible(true);

        return new ChartComponents(chart, dataset);
    }

    private static void configureAndShowProgressFrame(
            final List<Classifier> classifiers,
            final JTable progressTable,
            final Map<Integer, Boolean> visibleCurves,
            final Map<Integer, Integer> donePositions,
            final JFreeChart chart) {
        fillTableWithClassifiers(classifiers, progressTable,
                TABLE_HEADER_LABELS.length, visibleCurves);
        fitTableByWidth(progressTable, TABLE_COLUMN_MARGIN);

        final JFrame jf = prepareProgressFrame(progressTable, PROGRESS_FRAME_TITLE);
        jf.setVisible(true);

        progressTable.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(final MouseEvent e) {
                final JTable jTable = (JTable) e.getSource();
                final Point p = e.getPoint();
                final int row = jTable.rowAtPoint(p);

                if (e.getClickCount() >= 2 && donePositions.containsKey(row)) {
                    final XYItemRenderer renderer = ((XYPlot) chart.getPlot()).getRenderer();
                    final int plotNumber = donePositions.get(row);
                    final boolean status = !visibleCurves.get(plotNumber);
                    visibleCurves.put(plotNumber, status);
                    renderer.setSeriesVisible(plotNumber, status);
                }
            }
        });
    }

    private static JFrame prepareProgressFrame(
            final JTable table,
            final String progressFrameTitle) {
        final JFrame jf = new JFrame(progressFrameTitle);
        final JScrollPane scrollPane = new JScrollPane(table);
        final Dimension size = table.getPreferredSize();

        jf.getContentPane().setLayout(new BorderLayout());
        table.getTableHeader().setResizingAllowed(false);
        table.setBorder(BorderFactory.createEmptyBorder());

        scrollPane.setBorder(BorderFactory.createEmptyBorder());
        jf.getContentPane().add(scrollPane, BorderLayout.CENTER);
        jf.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        scrollPane.setPreferredSize(new Dimension(size.width - 10, size.height + 10));
        jf.setLocation(getCentralizedLocation(SCREEN_SIZE, jf.getSize()));

        jf.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(final WindowEvent e) {
                jf.dispose();
                System.exit(0);
            }
        });

        jf.pack();
        jf.setResizable(false);

        return jf;
    }

    private static void performClassification(
            final Instances data,
            final List<Classifier> classifiers,
            final ChartComponents chartComponents,
            final JTable table,
            final Map<Integer, Integer> donePositions) {
        final ExecutorService executor = Executors.newWorkStealingPool();

        classifiers.parallelStream()
                .map(classifier -> executor.submit(new ClassificationCallable(classifier, data)))
                .forEach(future -> processResult(future, chartComponents.getDataset(), chartComponents.getChart(),
                        table, classifiers, donePositions));

        executor.shutdown();
        System.out.println("Classifying complete");
    }

    private static JFreeChart prepareLineChart(final XYDataset dataset) {
        return ChartFactory.createXYLineChart(
                MAIN_FRAME_TITLE,
                HORIZONTAL_AXIS_LABEL, VERTICAL_AXIS_LABEL,
                dataset, PlotOrientation.VERTICAL,
                ENABLE_LEGEND, ENABLE_TOOLTIPS, NO_URLS);
    }

    private static void preparePlot(
            final XYPlot plot,
            final Paint backgroundColor,
            final Paint gridColor,
            final double gridStep) {
        final NumberAxis range = (NumberAxis) plot.getRangeAxis();
        final NumberAxis domain = (NumberAxis) plot.getDomainAxis();

        plot.setBackgroundPaint(backgroundColor);
        plot.setRangeGridlinePaint(gridColor);
        plot.setDomainGridlinePaint(gridColor);

        domain.setRange(ROC_LOWER_BOUND, ROC_UPPER_BOUND);
        domain.setTickUnit(new NumberTickUnit(gridStep));
        domain.setVerticalTickLabels(true);

        range.setRange(ROC_LOWER_BOUND, ROC_UPPER_BOUND);
        range.setTickUnit(new NumberTickUnit(gridStep));
        plot.setDomainGridlinesVisible(true);
    }

    private static Point getCentralizedLocation(
            final Dimension screenDimensions,
            final Dimension windowDimensions) {
        return new Point(
                screenDimensions.width / 2 - windowDimensions.width / 2,
                screenDimensions.height / 2 - windowDimensions.height / 2);
    }

    private static JFrame prepareMainFrame(
            final ChartPanel panel,
            final String title,
            final int width,
            final int height) {
        final JFrame jf = new JFrame(title);

        jf.setResizable(false);
        jf.setSize(width, height);
        jf.getContentPane().setLayout(new BorderLayout());
        jf.getContentPane().add(panel, BorderLayout.CENTER);
        jf.setLocation(getCentralizedLocation(SCREEN_SIZE, jf.getSize()));

        jf.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(final WindowEvent e) {
                jf.dispose();
                System.exit(0);
            }
        });

        return jf;
    }

    private static void fillTableWithClassifiers(
            final List<Classifier> classifiers,
            final JTable table,
            final int columnNumber,
            final Map<Integer, Boolean> visibleCurves) {
        IntStream.range(0, classifiers.size()).forEach(i -> {
            table.setValueAt(classifiers.get(i).getClass().getSimpleName(), i, 0);
            IntStream.range(1, columnNumber).forEach(j ->
                    table.setValueAt(INITIAL_PROGRESS_LABEL, i, j));
            visibleCurves.put(i, true);
        });
    }

    private static void fitTableByWidth(final JTable table, final int margin) {
        table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);

        IntStream.range(0, table.getColumnCount()).forEach(i -> {
            alignColumnToCenter(table, i);
            packTableColumn(table, i, margin);
        });

        alignColumnToLeft(table, FIRST_COLUMN);
    }

    private static void alignColumnToCenter(final JTable table, final int column) {
        final DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(SwingConstants.CENTER);
        table.getColumnModel().getColumn(column).setCellRenderer(centerRenderer);
    }

    private static void packTableColumn(final JTable table, final int column, final int margin) {
        final TableColumnModel colModel = table.getColumnModel();
        final TableColumn col = colModel.getColumn(column);

        final TableCellRenderer headerRenderer = col.getHeaderRenderer() == null
                ? table.getTableHeader().getDefaultRenderer()
                : col.getHeaderRenderer();

        final Component firstCellComponent = headerRenderer.getTableCellRendererComponent(table,
                col.getHeaderValue(), DOES_NOT_SELECTED, HAS_NO_FOCUS, HEADER_ROW, FIRST_COLUMN);
        int width = firstCellComponent.getPreferredSize().width;

        for (int row = 0; row < table.getRowCount(); row++) {
            final TableCellRenderer cellRenderer = table.getCellRenderer(row, column);
            final Component component = cellRenderer.getTableCellRendererComponent(table,
                    table.getValueAt(row, column), DOES_NOT_SELECTED, HAS_NO_FOCUS, row, column);
            width = Math.max(width, component.getPreferredSize().width);
        }

        width += 2 * margin;
        col.setPreferredWidth(width);
    }

    private static void alignColumnToLeft(final JTable table, final int column) {
        final DefaultTableCellRenderer centerRenderer = new DefaultTableCellRenderer();
        centerRenderer.setHorizontalAlignment(SwingConstants.LEFT);
        table.getColumnModel().getColumn(column).setCellRenderer(centerRenderer);
    }

    private static void processResult(
            final Future<ClassificationResult> future,
            final DefaultXYDataset dataset,
            final JFreeChart chart,
            final JTable table,
            final List<Classifier> classifiers,
            final Map<Integer, Integer> classifierPositions) {
        try {
            final ClassificationResult result = future.get();
            final int donePosition;

            synchronized (Main.class) {
                final int classifierIndex = classifiers.indexOf(result.getClassifier());
                donePosition = classifierPositions.size();
                classifierPositions.put(classifierIndex, classifierPositions.size());
            }

            addPlotToGraph(result, dataset, chart);
            updateProgressInformation(result, table, donePosition);
        } catch (Exception e) {
            throw new IllegalArgumentException(e);
        }
    }

    private static void addPlotToGraph(
            final ClassificationResult data,
            final DefaultXYDataset dataset,
            final JFreeChart chart) {
        final int plotNumber = dataset.getSeriesCount();
        final XYItemRenderer renderer = ((XYPlot) chart.getPlot()).getRenderer();

        final double[][] points = {
                prepareArrayForPlot(data.getFalsePositives()),
                prepareArrayForPlot(data.getTruePositives())
        };

        dataset.addSeries(data.getClassifier().getClass().getSimpleName(), points);
        renderer.setSeriesStroke(plotNumber, prepareStrokeByPlotNumber(plotNumber));
        chart.fireChartChanged();
    }

    private static String format(final double value) {
        return String.format("%5.3f", value);
    }

    private static void updateProgressInformation(
            final ClassificationResult result,
            final JTable table,
            final int donePosition) {
        table.setValueAt(format(result.getAreaUnderCurve()), donePosition, 1);
        table.setValueAt(format(result.getTruePositivesRate()), donePosition, 2);
        table.setValueAt(format(result.getTrueNegativesRate()), donePosition, 3);
        table.setValueAt(format(result.getMatthewsCorrelationCoefficient()), donePosition, 4);
        table.setValueAt(Long.toString(result.getExecutionTime()), donePosition, 5);
    }

    private static double[] prepareArrayForPlot(final double[] source) {
        final int n = source.length;
        final double max = Doubles.max(source);
        final double[] result = new double[n];

        for (int i = 0; i < n; i++) {
            result[i] = source[n - i - 1] / max;
        }

        return result;
    }

    private static Stroke prepareStrokeByPlotNumber(final int plotNumber) {
        return new BasicStroke(STROKE_WIDTH, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND,
                STROKE_MITERLIMIT, getStrokeDashByPlotNumber(plotNumber + 1), STROKE_DASH_PHASE);
    }

    @SuppressWarnings("NumericCastThatLosesPrecision")
    private static float[] getStrokeDashByPlotNumber(final int plotNumber) {
        return new float[]{(float) StrictMath.pow(plotNumber, DASH_GROW_POW) * 2};
    }

    private static class ChartComponents {
        JFreeChart chart;
        DefaultXYDataset dataset;

        ChartComponents(final JFreeChart chart, final DefaultXYDataset dataset) {
            this.chart = chart;
            this.dataset = dataset;
        }

        JFreeChart getChart() {
            return chart;
        }

        DefaultXYDataset getDataset() {
            return dataset;
        }
    }
}