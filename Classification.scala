//Breinda el parametro treeStrategy para el algoritmo del arbol ayudando al booster de la clasificacion binaria y
//numIterations Número de iteraciones de impulso. En otras palabras, el número de hipótesis débiles utilizadas en el modelo final. 
import org.apache.spark.mllib.tree.GradientBoostedTrees 
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils

// Load and parse the data file.
val data = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")
// Split the data into training and test sets (30% held out for testing)
val splits = data.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))
//--------------------------------------------------------------------------
// Entrena un modelo GradientBoostedTrees. 
// Los parámetros predeterminados para la clasificación usan LogLoss de forma predeterminada. 
val boostingStrategy = BoostingStrategy.defaultParams("Classification")
boostingStrategy.numIterations = 3 // Nota: use más iteraciones en la práctica. //Usando GradientBoostedTrees numIterations = int
boostingStrategy.treeStrategy.numClasses = 2//Usando GradientBoostedTrees 
boostingStrategy.treeStrategy.maxDepth = 15//Usando GradientBoostedTrees 

// Empty categoricalFeaturesInfo indica que todas las características son continuas. 
boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()//Usando GradientBoostedTrees 

val model = GradientBoostedTrees.train(trainingData, boostingStrategy)

// Evalúa el modelo en instancias de prueba y calcula el error de prueba 
val labelAndPreds = testData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}
val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
//Muestra el error
println(s"Test Error = $testErr")
//Despliega todo lo que aprendio el algoritmo desplegandolo en 3 arboles indicado en la linea 17
println(s"Learned classification GBT model:\n ${model.toDebugString}")

// Guardar y cargar modelo 
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")

val sameModel = GradientBoostedTreesModel.load(sc,
  "target/tmp/myGradientBoostingClassificationModel")
/*
treeStrategy Parámetros para el algoritmo del árbol. Apoyamos la regresión y la clasificación binaria para impulsar. 
La configuración de impurezas será ignorada. 
param: pérdida Función de pérdida utilizada para la minimización durante el aumento de gradiente. 
param: numIterations Número de iteraciones de impulso. En otras palabras, el número de hipótesis débiles utilizadas en el modelo final. 
param: learningRate Tasa de aprendizaje para reducir la contribución de cada estimador. La tasa de aprendizaje debe estar entre el intervalo (0, 1] 
param: validationTol Útil cuando se utiliza runWithValidation. Si la tasa de error en la entrada de validación entre dos iteraciones es menor que validationTol, entonces se detiene. 
Ignorado cuando runse usa.*/