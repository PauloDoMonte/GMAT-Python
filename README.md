# GMAT-Python
Simulações de missão espaciais realizadas no GMAT com suporte a linguagem python para resolução de problemas como mitigação espacial de neo ou planejamento de manobras espaciais como rendezvous

## Rendezvous
Primeiro iremos realizar a busca da primeira propulsão, que será responsavél por fazer nossa Nave se aproximar do Detrito que está em uma orbita mais alta. O Algoritimo génetico irá procurar parametros que resulte em uma distancia relativa que tenda a zero, após encontrar essa propulsão, iremos aplicar ela e o proximo passo será encontrar uma segunda propulsão que será responsavél por circularizar a orbita da Nave ao ponto de ficar coincidente com a orbita do Detrito, ela irá fazer isso, buscando a distância e velocidade relativa proxima a zero, porque assim, irá conseguir se aproximar mais e permanecer na mesma orbita, sendo assim, realizando o Rendezvous.
