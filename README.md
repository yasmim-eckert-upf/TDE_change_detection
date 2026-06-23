# Detecção de Mudanças Urbanas com Siamese U-Net

Solução de visão computacional que identifica, de forma automática, **o que mudou** entre duas imagens de satélite do mesmo local capturadas em momentos diferentes (ex.: construção ou demolição de edificações). O sistema gera uma máscara de segmentação destacando as regiões alteradas e oferece uma interface gráfica para uso com imagens próprias.

## Equipe

- Fernanda Ihjaz
- Yasmim Eckert Ferri

## Demonstração em vídeo

📹 **[Assista à demonstração da ferramenta aqui](https://app.betterbugs.io/session/6a3ae85cd3521c318341827a?openedDevTab=info&openedNetworkSubTab=all&openedConsoleSubTab=all&openedStepsSubTab=all)**

## Motivação e aplicabilidade

A detecção automática de mudanças (*change detection*) a partir de imagens de satélite tem aplicações diretas em:

- planejamento e fiscalização urbana (construções irregulares, expansão urbana);
- gestão de desastres (mapeamento de áreas afetadas por enchentes, deslizamentos);
- monitoramento ambiental (desmatamento, alterações de cobertura do solo).

A partir de um par de imagens (antes/depois), o modelo devolve um mapa preciso das áreas que sofreram alteração.

## Como funciona

A solução combina **processamento de imagens com OpenCV** e uma rede neural profunda do tipo **Siamese U-Net**.

### Arquitetura Siamese U-Net

A rede possui dois ramos *encoder* independentes (um para a imagem "antes", outro para a "depois") que extraem características de cada imagem separadamente. A **diferença absoluta** entre as características dos dois ramos, em cada nível de resolução, é usada como *skip connection*, realçando exatamente onde houve mudança. Um *decoder* compartilhado reconstrói progressivamente a resolução original e produz a máscara binária final.

```
Imagem A (antes) ──> Encoder A ─┐
                                 ├──> |diferença| ──> Decoder ──> Máscara de mudança
Imagem B (depois) ─> Encoder B ─┘
```

### Técnicas de processamento de imagem (OpenCV)

O OpenCV é utilizado em todo o pipeline de manipulação das imagens:

| Etapa | Técnica / Função |
|---|---|
| Leitura das imagens | `cv2.imread` |
| Conversão de espaço de cor | `cv2.cvtColor` (BGR → RGB e leitura das máscaras em escala de cinza) |
| Normalização | conversão para `float32` e escala para o intervalo `[0, 1]` |
| Redimensionamento | ajuste para 256×256 pixels |
| Augmentação de dados | flips horizontal/vertical e rotações sincronizados entre imagem e máscara |
| Segmentação | geração da máscara binária a partir da saída da rede |
| Pós-processamento / visualização | redimensionamento da máscara e destaque das mudanças em vermelho sobre a imagem |

## Interface gráfica (GUI)

A interface é construída com **Gradio**. O usuário:

1. faz o upload de duas imagens (pré-mudança e pós-mudança);
2. clica em **"Detectar Mudanças"**;
3. recebe duas saídas: a **máscara binária** das mudanças e um **overlay** que destaca as áreas alteradas em vermelho sobre a imagem.

A GUI aceita **imagens de entrada quaisquer** (não apenas as do conjunto de teste): as imagens são redimensionadas e normalizadas automaticamente antes da inferência, e a máscara resultante é reescalada para o tamanho original da imagem enviada. A pasta `imagens_para_interface/` contém exemplos prontos para testar a interface.

## Dataset

O modelo foi treinado no **LEVIR-CD**, um conjunto público de 637 pares de imagens de satélite de alta resolução (1024×1024) com máscaras de mudança anotadas.

🔗 Download: https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd

Baixe o arquivo `.zip` (~2 GB), descompacte na pasta do projeto e renomeie a pasta para `LEVIR_CD`. A estrutura esperada é:

```
LEVIR_CD/
├── train/
│   ├── A/       # imagens "antes"
│   ├── B/       # imagens "depois"
│   └── label/   # máscaras de mudança (ground truth)
├── val/
│   ├── A/  
    ├── B/  
    └── label/
└── test/
    ├── A/  
    ├── B/  
    └── label/
```

**Observação:** o dataset não é versionado no repositório (está no `.gitignore`) e deve ser obtido separadamente. Caso o coloque em outro local, ajuste a variável `root` no notebook:
```python
root = r"caminho-da-pasta\LEVIR_CD"
```

## Resultados

Treinamento por 50 épocas. 
Métricas finais no conjunto de teste:

| IoU | Dice | Precisão | Recall | Acurácia | Loss |
|---|---|---|---|---|---|
| 0,69 | 0,82 | 0,81 | 0,82 | 0,98 | 0,11 |

A evolução das curvas de treino/validação está em `training_evolution.png`, e exemplos de predição em `test_result_1.png` e `test_result_2.png`.

## Estrutura do repositório

```
.
├── tde_change_detection.ipynb   # notebook principal (dataset, modelo, treino, GUI)
├── imagens_para_interface/      # imagens de exemplo para testar a interface
├── requirements.txt             # dependências do projeto
├── README.md
└── .gitignore
```

### Arquivos gerados pela execução do notebook

Ao rodar o notebook, são criados os seguintes arquivos (resultados):

| Arquivo | Descrição |
|---|---|
| `best_model.pt` | melhor modelo treinado (maior IoU de validação) |
| `last_checkpoint.pt` | checkpoint da última época |
| `train_log.txt` | log de métricas por época |
| `test_log.txt` | métricas finais no conjunto de teste |
| `training_evolution.png` | curvas de Loss / IoU / Dice |
| `test_result_1.png`, `test_result_2.png` | exemplos de predição |

> Os pesos do modelo (`*.pt`), a pasta `LEVIR_CD/` estão no `.gitignore` por serem grandes.

> Gradio (`.gradio/`) e (`*.txt`) ou (`*.png`) gerados estão no `.gitignore` não fazerem parte do código-fonte. É necessário executar para que sejam gerados e a GUI funcione.

## Como executar

### Pré-requisitos

- Python 3.12
- GPU NVIDIA com CUDA ou CPU (mais lento)
- O dataset LEVIR-CD baixado e organizado conforme a seção [Dataset](#dataset)

### 1. Ambiente virtual e dependências

No terminal, dentro da pasta do projeto:

```powershell
# cria e ativa o ambiente virtual
py -3.12 -m venv venv
.\venv\Scripts\activate

# instala o PyTorch com suporte a CUDA (ajuste a versão de CUDA conforme sua GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# instala as demais dependências
pip install -r requirements.txt
```

### 2. Registrar o kernel do ambiente virtual

```powershell
python -m pip install --upgrade pip
python -m pip install ipykernel
python -m ipykernel install --user --name venv --display-name "Python (venv)"
```

### 3. Selecionar kernel

No arquivo `tde_change_detection.ipynb`, clique em **Select Kernel** -> **Python Environments...** -> selecione **`venv (Python 3.12.0)`** (o ambiente virtual, *não* o Python global).

### 4. Executar

- **Para treinar do zero:** execute as células de cima para baixo, em ordem. O treinamento gera `best_model.pt`, `last_checkpoint.pt`, os logs e as imagens de resultado.
- **Para apenas usar a GUI** (sem retreinar): execute, em ordem, as células de *imports* e de *arquitetura do modelo*, defina `CARREGAR_MODELO = True` na célula de carregamento e execute-a (carrega o `best_model.pt`). Em seguida, execute a célula da **interface Gradio**.

## Tecnologias utilizadas

- **OpenCV** — leitura, conversão de cores, redimensionamento e composição de imagens
- **PyTorch** — definição e treinamento da rede Siamese U-Net
- **Albumentations** — pipeline de augmentação sincronizada entre imagens e máscaras
- **Gradio** — interface gráfica para inferência
- **NumPy / Matplotlib** — manipulação numérica e visualização