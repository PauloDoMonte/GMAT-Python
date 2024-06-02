# GMAT-Python
Simulações de missão espaciais realizadas no GMAT com suporte a linguagem python para resolução de problemas como mitigação espacial de neo ou planejamento de manobras espaciais como rendezvous

## Redes Neurais
Este sistema implementa um modelo de rede neural para prever a trajetória de objetos em órbita espacial. Através da normalização de dados e do treinamento de uma rede neural complexa (MLP) com Sklearn e Tensorflow, o sistema aprende a identificar padrões nos dados de entrada e a prever a trajetória do objeto ao longo do tempo. A avaliação em um conjunto de dados de teste garante a precisão do modelo, que pode ser aplicado em diversas áreas, como monitoramento espacial, controle de tráfego orbital e simulação de missões espaciais.

## Rendezvous
Primeiro iremos realizar a busca da primeira propulsão, que será responsavél por fazer nossa Nave se aproximar do Detrito que está em uma orbita mais alta. O Algoritimo génetico irá procurar parametros que resulte em uma distancia relativa que tenda a zero, após encontrar essa propulsão, iremos aplicar ela e o proximo passo será encontrar uma segunda propulsão que será responsavél por circularizar a orbita da Nave ao ponto de ficar coincidente com a orbita do Detrito, ela irá fazer isso, buscando a distância e velocidade relativa proxima a zero, porque assim, irá conseguir se aproximar mais e permanecer na mesma orbita, sendo assim, realizando o Rendezvous.
