# Detecção de Mudanças Urbanas com Siamese U-Net

> Solução de visão computacional que identifica, de forma automática, **o que mudou** entre duas imagens de satélite do mesmo local capturadas em momentos diferentes (ex.: construção ou demolição de edificações). O sistema gera uma máscara de segmentação destacando as regiões alteradas e oferece uma interface gráfica para uso com imagens próprias.

Trabalho prático da disciplina **CCC309 – Processamento de Imagens e Visão Computacional** (2026/1) — Universidade de Passo Fundo.

## Equipe

- Fernanda Japur Ihjaz — 205657@upf.br
- Yasmim Eckert Ferri — 183395@upf.br

## Demonstração em vídeo

📹 **[Assista à demonstração da ferramenta aqui](#)** *(o vídeo ainda não foi gravado)*

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

## Interface gráfica (GUI)

## Dataset

O dataset pode ser obtido em:
https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd

## Resultados

## Estrutura do repositório

```
.
│
├── tde.ipynb                    # notebook principal (dataset, modelo, treino, GUI)
│
├── train_log.txt                # log de métricas por época
│
├── test_log.txt                 # métricas finais no conjunto de teste
│
├── evolucao_treinamento.png     # curvas de Loss / IoU / Dice
│
├── resultado_teste_1.png        # exemplos de predição
│
├── resultado_teste_2.png
│
├── requirements.txt             # dependências do projeto
│
├── .gitignore
│
└── LEVIR_CD/                    # dataset
    ├── train/
    │   ├── A/       # imagem pré-mudança
    │   ├── B/       # imagem pós-mudança
    │   └── label/   # máscara de mudança
    │
    ├── val/
    └── test/
```

## Como executar

### Pré-requisitos

- Python 3.11
- GPU NVIDIA com CUDA (recomendado; o projeto roda em CPU, porém mais lentamente)
- O dataset LEVIR-CD organizado conforme a estrutura acima

### Instalação

```bash
# 1. Crie e ative um ambiente virtual
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# 2. Instale o PyTorch com suporte a CUDA (ajuste a versão de CUDA conforme sua GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Instale as demais dependências
pip install -r requirements.txt
```

### Execução

1. No `tde.ipynb`, ajuste a variável `root` para o caminho da pasta `LEVIR_CD` na sua máquina.
2. Abra o notebook:
   ```bash
   jupyter notebook
   ```
3. **Para apenas usar a GUI** (sem retreinar): execute, em ordem, as células de *imports* e de *arquitetura do modelo*, defina `CARREGAR_MODELO = True` na célula de carregamento e execute-a (carrega o `best_model.pt`). Em seguida, execute a célula da **interface Gradio**.
4. **Para treinar do zero:** execute as células de cima para baixo, em ordem. O treinamento gera `best_model.pt` e `last_checkpoint.pt`.

## Tecnologias utilizadas

- **OpenCV** — leitura, conversão de cores, redimensionamento e composição de imagens
- **PyTorch** — definição e treinamento da rede Siamese U-Net
- **Albumentations** — pipeline de augmentação sincronizada entre imagens e máscaras
- **Gradio** — interface gráfica para inferência
- **NumPy / Matplotlib** — manipulação numérica e visualização