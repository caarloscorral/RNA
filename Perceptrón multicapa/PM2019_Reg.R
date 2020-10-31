library(RSNNS)


## funcion que calcula el error cuadratico medio
MSE <- function(pred,obs) {sum((pred-obs)^2)/length(obs)}


#CARGA DE DATOS
# se supone que los ficheros tienen encabezados
trainSet <- read.csv("Training.csv",dec=".",sep=",",header = T)
validSet <- read.csv( "Validation.csv",dec=".",sep=",",header = T)
testSet  <- read.csv("Testing.csv",dec=".",sep=",",header = T)

salida <- ncol (trainSet)   #num de la columna de salida


topologias <- list()
topologias[[1]] <- c(30)
topologias[[2]] <- c(10,10)
topologias[[3]] <- c(10,20)
topologias[[4]] <- c(10,30)
topologias[[5]] <- c(15,30)
topologias[[6]] <- c(20,10)
topologias[[7]] <- c(20,20)
topologias[[8]] <- c(20,30)
topologias[[9]] <- c(30,15)
topologias[[10]] <- c(30,15,30)
topologias[[11]] <- c(10,10,10)
topologias[[12]] <- c(10,20,10)
topologias[[13]] <- c(10,30,10)
topologias[[14]] <- c(15,30,15)

razones <- c(0.001, 0.005, 0.01, 0.02, 0.03, 0.1, 0.5)

for (top in topologias) {
        for (razon in razones) {
                #SELECCION DE LOS PARAMETROS
                topologia        <- top #PARAMETRO DEL TIPO c(A,B,C,...,X) A SIENDO LAS NEURONAS EN LA CAPA OCULTA 1, B LA CAPA 2 ...
                razonAprendizaje <- razon #NUMERO REAL ENTRE 0 y 1
                ciclosMaximos    <- 10000 #NUMERO ENTERO MAYOR QUE 0
                
                #EJECUCION DEL APRENDIZAJE Y GENERACION DEL MODELO
                
                set.seed(1)
                model <- mlp(x= trainSet[,-salida],
                             y= trainSet[, salida],
                             inputsTest=  validSet[,-salida],
                             targetsTest= validSet[, salida],
                             size= topologia,
                             maxit=ciclosMaximos,
                             learnFuncParams=c(razonAprendizaje),
                             shufflePatterns = F
                             )
                
                # #GRAFICO DE LA EVOLUCION DEL ERROR
                plotIterativeError(model)
                # 
                # # DATAFRAME CON LOS ERRORES POR CICLo: de entrenamiento y de validacion
                # iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                #                               MSEValid= (model$IterativeTestError/nrow(validSet)))
                # 
                # # 
                 #SE OBTIENE EL N?MERO DE CICLOS DONDE EL ERROR DE VALIDACION ES MINIMO 
                 nuevosCiclos <- which.min(model$IterativeTestError)
                # 
                 #ENTRENAMOS LA MISMA RED CON LAS ITERACIONES QUE GENERAN MENOR ERROR DE VALIDACION
                 set.seed(1)
                 model <- mlp(x= trainSet[,-salida],
                             y= trainSet[, salida],
                             inputsTest=  validSet[,-salida],
                             targetsTest= validSet[, salida],
                             size= topologia,
                             maxit=nuevosCiclos,
                             learnFuncParams=c(razonAprendizaje),
                             shufflePatterns = F
                )
                #GRAFICO DE LA EVOLUCION DEL ERROR
                jpeg(file=paste("Outputs/nnet",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".jpg", sep = ""))
                plotIterativeError(model)
                dev.off()
                print("Hey")
                iterativeErrors <- data.frame(MSETrain= (model$IterativeFitError/ nrow(trainSet)),
                                              MSEValid= (model$IterativeTestError/nrow(validSet)))
                
                #CALCULO DE PREDICCIONES
                prediccionesTrain <- predict(model,trainSet[,-salida])
                prediccionesValid <- predict(model,validSet[,-salida])
                prediccionesTest  <- predict(model, testSet[,-salida])
                
                #CALCULO DE LOS ERRORES
                errors <- c(TrainMSE= MSE(pred= prediccionesTrain,obs= trainSet[,salida]),
                            ValidMSE= MSE(pred= prediccionesValid,obs= validSet[,salida]),
                            TestMSE=  MSE(pred= prediccionesTest ,obs=  testSet[,salida]))
                errors
                
                
                
                
                
                #SALIDAS DE LA RED
                outputsTrain <- data.frame(pred= prediccionesTrain,obs= trainSet[,salida])
                outputsValid <- data.frame(pred= prediccionesValid,obs= validSet[,salida])
                outputsTest  <- data.frame(pred= prediccionesTest, obs=  testSet[,salida])
                
                
                
                
                #GUARDANDO RESULTADOS
                saveRDS(model,paste("Outputs/nnet",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".rds", sep = "", collapse = NULL))
                write.csv2(errors,paste("Outputs/finalErrors",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".csv", sep = "", collapse = NULL))
                write.csv2(iterativeErrors,paste("Outputs/iterativeErrors",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".csv", sep = "", collapse = NULL))
                write.csv2(outputsTrain,paste("Outputs/netOutputsTrain",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".csv", sep = "", collapse = NULL))
                write.csv2(outputsValid,paste("Outputs/netOutputsValid",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".csv", sep = "", collapse = NULL))
                write.csv2(outputsTest, paste("Outputs/netOutputsTest",paste(paste(top,sep="", collapse = "-"), razon, sep="-"), ".csv", sep = "", collapse = NULL))
}
}

