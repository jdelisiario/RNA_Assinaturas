#limpar workspace
rm(list=ls())

#limpar tela
cat('\014')
library("Rcpp")
#bibliotecas
library("RSNNS")

#usada para plotar graficos do tipo scatter em 3d
#library("rgl")

#usada para gerar matrix de confusao
#library("SDMTools")

#---------------------------------------
#FUNCOES

#funcao que ajusta outliers de qualquer amostra
ajustaOutliers <- function(x, na.rm = TRUE, ...) 
{
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  
  for(i in 1:length(y)) 
  {
    #caso o primeiro valor seja NA procura o proximo valor nao NA e coloca
    #no lugar do NA
    if (is.na(y[1]) == TRUE)
    {
      encontrou = FALSE
      cont = 1
      posterior = NA
      #procura o primeiro numero POSTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[1+cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          posterior <- y[1+cont];
          encontrou <- TRUE
        }
      }
      
      y[1] <- posterior
    }
    
    #caso o ultimo valor seja NA procura o primeiro valor anterior que nao NA e coloca
    #no lugar do NA
    if (is.na(y[length(y)]) == TRUE)
    {
      encontrou <- FALSE
      cont <- 1
      anterior <- NA
      
      #procura o primeiro numero ANTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[length(y)-cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          anterior <- y[length(y)-cont];
          encontrou <- TRUE
        }
      }
      
      y[length(y)] <- anterior
    }
    
    
    
    if (is.na(y[i])==TRUE)
    {
      encontrou <- FALSE
      cont <- 1
      anterior <- NA
      
      #procura o primeiro numero ANTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[i-cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          anterior <- y[i-cont];
          encontrou <- TRUE
        }
      }
      
      encontrou = FALSE
      cont = 1
      posterior = NA
      
      #procura o primeiro numero POSTERIOR ao valor atual que nao seja NA
      while (encontrou == FALSE)
      {
        if (is.na(y[i+cont]) == TRUE)
        {
          cont <- cont + 1
        }
        else
        {
          posterior <- y[i+cont];
          encontrou <- TRUE
        }
      }
      
      #executa uma media entre o anterior e posterior valor valido na serie e insere no lugar do outlier
      y[i] <- (anterior+posterior)/2
    }
  }
  
  return(y)
}

#Coloca amostra nos intervalos proporcionais entre 0 e 1
padroniza <- function(s)
{
  retorno <- (s - min(s))/(max(s)-min(s))
  return(retorno)
}


carrega_subset <- function(arquivo) 
{
  dados <- read.table(arquivo,
                      header=TRUE,
                      sep=";",
                      colClasses=c("character", rep("numeric",6)),
                      na="?")
  return(dados)
}

plotaDivisoes <- function()
{
  abline(h = 0, v = 10.5, col = "red", lty=2)
  abline(h = 0, v = 20.5, col = "red", lty=2)
  abline(h = 0, v = 30.5, col = "red", lty=2)
}

defineClasse <- function(v1,v2)
{
  if (v1==0 & v2==0){
    return(1)
  }
  else if (v1==0 & v2==1){
      return(2)
  }
  else if (v1==1 & v2==0){
      return(3)
  }
  else
      return (4)
}
#----------------------------------------------------------------
#Carregamento do dataset

treinamento <- carrega_subset("treinamento.csv")
#separação das entradas
contorno = treinamento$qt_pixels_contorno
perimetro = treinamento$perimetro
casco = treinamento$qt_pixels_hull
areaRet = treinamento$area_ret_circuns


plot(contorno, main="Contorno")
plotaDivisoes()

plot(perimetro, main="Perimetro")
plotaDivisoes()

plot(casco, main="Casco")
plotaDivisoes()

#scatter 3D
cores <- c(rep(1,10),rep(2,10),rep(3,10),rep(4,10))
dados <- cbind(contorno,perimetro,casco,cores)
#plot3d(dados[,1:3], col=dados[,4], size=8)

#boxplot
boxplot(contorno, main="Contorno")
boxplot(perimetro, main="Perimetro")
boxplot(casco, main="Casco")

#Padronização das séries
x1 <- padroniza(as.numeric(contorno))
x2 <- padroniza(as.numeric(perimetro))
x3 <- padroniza(as.numeric(casco))

x <- cbind(x1,x2,x3)

y<- cbind(treinamento$quem,treinamento$quem2)

#IMPLEMENTACAO DA REDE NEURAL NOS DADOS AJUSTADOS
#configuracoes da MLP
nNeuronios = 20
maxEpocas <- 100000

#treinamento da MLP
rede <- NULL
set.seed(1)
#Backpropagation
print("treinando a rede na serie ajustada...")

rede<-mlp(x, y, size=nNeuronios, maxit=maxEpocas, initFunc="Randomize_Weights",
            initFuncParams=c(-0.3, 0.3), learnFunc="Std_Backpropagation",
            learnFuncParams=c(0.027), updateFunc="Topological_Order",
            updateFuncParams=c(0), hiddenActFunc="Act_Logistic",
            shufflePatterns=T, linOut=TRUE)
save(rede, file='Rede_Assinaturas.Rdata')

#load('Rede_Assinaturas.Rdata')

plot(rede$IterativeFitError,type="l",main="Evolucao do EQM")

print(rede$IterativeFitError[length(rede$IterativeFitError)])

#AVALIANDO PREVISOES NO CONJUNTO DE TREINAMENTO
y1hat_treino = vector()
y2hat_treino = vector()

for (i in 1:length(x[,1]))
{
  y1hat_treino[i] = predict(rede,x[i,])[1]
  y2hat_treino[i] = predict(rede,x[i,])[2]
}

yhat_treino <- cbind(y1hat_treino,y2hat_treino)

#CALCULO DO ERRO
erro_treino <- mean(sqrt((y-yhat_treino)^2))
print(paste("Erro no conjunto de treino:",erro_treino))


#MATRIZ DE CONFUSAO
#print("MATRIZ DE CONFUSAO NO TREINO")
#confusao <- confusion.matrix(y,yhat_treino)
#print(confusao)

#EXECUTANDO A PREVISOES COM O MODELO TREINADO
#Carregando o dataset de testes
validacao <- carrega_subset("validacao.csv")

contorno_validacao <- padroniza(as.numeric(validacao$qt_pixels_contorno))
perimetro_validacao <- padroniza(as.numeric(validacao$perimetro))
casco_validacao <- padroniza(as.numeric(validacao$qt_pixels_hull))

x_validacao <- cbind(contorno_validacao, perimetro_validacao, casco_validacao)
y_validacao <-cbind(validacao$quem,validacao$quem2)

y1hat_validacao = vector()
y2hat_validacao = vector()

for (i in 1:length(x_validacao[,1]))
{
  y1hat_validacao[i] = predict(rede,x_validacao[i,])[1]
  y2hat_validacao[i] = predict(rede,x_validacao[i,])[2]
}

yhat_validacao <- cbind(y1hat_validacao,y2hat_validacao)

#CALCULO DO ERRO
erro_validacao <- mean(sqrt((y_validacao-yhat_validacao)^2))
print(paste("Erro no conjunto de validacao:",erro_validacao))

yhat_validacao_ajustado <- cbind(round(y1hat_validacao), round(y2hat_validacao))

y_cores <- vector()
for (i in 1:length(y_validacao[,1]))
{
  y_cores[i] <- defineClasse(y_validacao[i,1],y_validacao[i,2])
}

yhat_cores_validacao <- vector()
for (i in 1:length(yhat_validacao_ajustado[,1]))
{
  yhat_cores_validacao[i] <- defineClasse(yhat_validacao_ajustado[i,1],yhat_validacao_ajustado[i,2])
}

vetor_acuracia = sqrt((y_cores-yhat_cores_validacao)^2)
print(paste("Acurácia no conjunto de validação: ", (length(which(vetor_acuracia == 0))*100)/length(vetor_acuracia), "%"))

#MATRIZ DE CONFUSAO
#print("MATRIZ DE CONFUSAO NA VALIDACAO")
#confusao <- confusion.matrix(y_validacao,yhat_validacao)
#print(confusao)

#PLOTANDO A AMOSTRA DESCONHECIDA CLASSIFICADA PELA REDE
#scatter 3D
#dados <- cbind(contorno_validacao,perimetro_validacao,casco_validacao)
#plot3d(dados[,1:3], col=yhat_cores_validacao, size=8)

#IMPLEMENTAR NO CONJUNTO DE TESTES

#EXECUTANDO A PREVISOES COM O MODELO TREINADO
#Carregando o dataset de testes
teste <- carrega_subset("teste.csv")

contorno_teste <- padroniza(as.numeric(teste$qt_pixels_contorno))
perimetro_teste <- padroniza(as.numeric(teste$perimetro))
casco_teste <- padroniza(as.numeric(teste$qt_pixels_hull))

x_teste <- cbind(contorno_teste, perimetro_teste, casco_teste)
y_teste <-cbind(teste$quem,teste$quem2)

y1hat_teste = vector()
y2hat_teste = vector()

for (i in 1:length(x_teste[,1]))
{
  y1hat_teste[i] = predict(rede,x_teste[i,])[1]
  y2hat_teste[i] = predict(rede,x_teste[i,])[2]
}

yhat_teste <- cbind(y1hat_teste,y2hat_teste)

#CALCULO DO ERRO
erro_teste <- mean(sqrt((y_teste-yhat_teste)^2))
print(paste("Erro no conjunto de teste:",erro_teste))

yhat_teste_ajustado <- cbind(round(y1hat_teste), round(y2hat_teste))

y_cores <- vector()
for (i in 1:length(y_teste[,1]))
{
  y_cores[i] <- defineClasse(y_teste[i,1],y_teste[i,2])
}

yhat_cores_teste <- vector()
for (i in 1:length(yhat_teste_ajustado[,1]))
{
  yhat_cores_teste[i] <- defineClasse(yhat_teste_ajustado[i,1],yhat_teste_ajustado[i,2])
}

vetor_acuracia_teste = sqrt((y_cores-yhat_cores_teste)^2)
print(paste("Acurácia no teste: ", (length(which(vetor_acuracia_teste == 0))*100)/length(vetor_acuracia_teste), "%"))
