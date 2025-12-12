# Guia de Uso - Experimento de Inversão de Embeddings

Este guia explica como executar o experimento completo de análise de riscos de reconstrução textual.

## 📋 Pré-requisitos

### 1. Ambiente Python

```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### 2. Dados

Certifique-se de que o arquivo `updated_dataset_preprocessed.parquet_new.gzip` está no diretório raiz do projeto.

## 🚀 Execução Rápida

### Executar Experimento Completo

```bash
python run_experiment.py
```

Este comando executará todas as 3 etapas do experimento:
1. Geração de embeddings
2. Treinamento do modelo inversor
3. Avaliação e geração de relatório

## ⚙️ Execução Personalizada

### Opções de Configuração

```bash
python run_experiment.py \
  --data_path updated_dataset_preprocessed.parquet_new.gzip \
  --max_samples 5000 \
  --model_type mlp \
  --batch_size 32 \
  --epochs 20 \
  --learning_rate 1e-4
```

### Parâmetros Disponíveis

| Parâmetro | Descrição | Padrão | Opções |
|-----------|-----------|--------|--------|
| `--data_path` | Caminho para o dataset | `updated_dataset_preprocessed.parquet_new.gzip` | - |
| `--max_samples` | Número máximo de amostras | `5000` | Qualquer inteiro |
| `--model_type` | Tipo de modelo inversor | `mlp` | `mlp`, `lstm`, `attention` |
| `--batch_size` | Tamanho do batch | `32` | Qualquer inteiro |
| `--epochs` | Número de épocas | `20` | Qualquer inteiro |
| `--learning_rate` | Taxa de aprendizado | `1e-4` | Qualquer float |
| `--eval_samples` | Amostras para avaliação | `None` (todas) | Qualquer inteiro |
| `--steps` | Etapas a executar | `1,2,3` | `1`, `2`, `3` ou combinações |
| `--force` | Forçar regeneração | `False` | Flag booleana |

### Executar Etapas Específicas

```bash
# Apenas gerar embeddings
python run_experiment.py --steps 1

# Apenas treinar modelo (requer embeddings)
python run_experiment.py --steps 2

# Apenas avaliar (requer modelo treinado)
python run_experiment.py --steps 3

# Treinar e avaliar
python run_experiment.py --steps 2,3
```

### Forçar Regeneração

```bash
# Regenerar tudo do zero
python run_experiment.py --force
```

## 📊 Tipos de Modelos

### 1. MLP (Multi-Layer Perceptron)
- **Mais rápido** para treinar
- Arquitetura simples
- Bom para experimentos iniciais

```bash
python run_experiment.py --model_type mlp
```

### 2. LSTM (Long Short-Term Memory)
- Captura dependências sequenciais
- Mais lento que MLP
- Melhor para textos longos

```bash
python run_experiment.py --model_type lstm
```

### 3. Attention (Transformer-based)
- **Mais sofisticado**
- Usa mecanismo de atenção
- Melhor performance, mas mais lento

```bash
python run_experiment.py --model_type attention
```

## 📁 Estrutura de Saída

Após a execução, os seguintes arquivos serão gerados:

```
.
├── data/
│   └── embeddings/
│       ├── train_embeddings.pkl      # Embeddings de treino
│       └── test_embeddings.pkl       # Embeddings de teste
│
├── models/
│   └── attacker/
│       └── {model_type}/
│           └── best_inverter.pt      # Melhor modelo treinado
│
└── results/
    ├── attack_metrics.json           # Métricas em JSON
    ├── ATTACK_REPORT.md              # Relatório completo
    ├── reconstruction_examples.txt   # Exemplos de reconstrução
    └── plots/
        ├── similarity_metrics.png    # Gráfico de similaridade
        ├── keyword_recovery.png      # Gráfico de recuperação
        └── risk_assessment.png       # Avaliação de risco
```

## 🔬 Execução Passo a Passo

### Etapa 1: Geração de Embeddings

```bash
python -m src.embedding.bertimbau_embedder
```

Ou usando o script principal:
```bash
python run_experiment.py --steps 1
```

**Saída esperada:**
- `data/embeddings/train_embeddings.pkl`
- `data/embeddings/test_embeddings.pkl`

### Etapa 2: Treinamento do Modelo Inversor

```bash
python -m src.attack.train_inverter
```

Ou usando o script principal:
```bash
python run_experiment.py --steps 2 --model_type mlp --epochs 20
```

**Saída esperada:**
- `models/attacker/mlp/best_inverter.pt`
- Logs de treinamento no terminal

### Etapa 3: Avaliação e Relatório

```bash
python -m src.evaluation.evaluate_attack
```

Ou usando o script principal:
```bash
python run_experiment.py --steps 3
```

**Saída esperada:**
- Relatório completo em `results/ATTACK_REPORT.md`
- Métricas em `results/attack_metrics.json`
- Visualizações em `results/plots/`

## 📈 Interpretação dos Resultados

### Métricas de Similaridade

- **BLEU (0-1)**: Mede sobreposição de n-gramas
  - > 0.5: Alta similaridade
  - 0.3-0.5: Similaridade moderada
  - < 0.3: Baixa similaridade

- **ROUGE (0-1)**: Mede recall de n-gramas
  - > 0.6: Boa recuperação
  - 0.4-0.6: Recuperação moderada
  - < 0.4: Baixa recuperação

### Recuperação de Palavras-Chave

- **Precision**: Proporção de palavras recuperadas corretas
- **Recall**: Proporção de palavras originais recuperadas
- **F1-Score**: Média harmônica de precision e recall

### Avaliação de Risco

- **Score < 0.3**: 🟢 Risco Baixo
- **Score 0.3-0.6**: 🟡 Risco Médio
- **Score > 0.6**: 🔴 Risco Alto

## 🐛 Solução de Problemas

### Erro: CUDA out of memory

```bash
# Reduzir batch size
python run_experiment.py --batch_size 16

# Ou usar CPU
export CUDA_VISIBLE_DEVICES=""
python run_experiment.py
```

### Erro: Arquivo não encontrado

```bash
# Verificar se o dataset existe
ls -lh updated_dataset_preprocessed.parquet_new.gzip

# Especificar caminho completo
python run_experiment.py --data_path /caminho/completo/para/dataset.gzip
```

### Treinamento muito lento

```bash
# Usar menos amostras
python run_experiment.py --max_samples 1000

# Usar modelo mais simples
python run_experiment.py --model_type mlp

# Reduzir épocas
python run_experiment.py --epochs 10
```

## 💡 Dicas de Uso

### Para Experimentos Rápidos

```bash
# Teste rápido com 1000 amostras
python run_experiment.py --max_samples 1000 --epochs 5
```

### Para Resultados de Produção

```bash
# Usar todas as amostras e mais épocas
python run_experiment.py --max_samples 50000 --epochs 30 --model_type attention
```

### Para Comparar Modelos

```bash
# Treinar todos os tipos de modelo
for model in mlp lstm attention; do
  python run_experiment.py --model_type $model --steps 2,3
done
```

## 📚 Próximos Passos

1. **Análise dos Resultados**: Revisar `results/ATTACK_REPORT.md`
2. **Ajuste de Hiperparâmetros**: Experimentar diferentes configurações
3. **Técnicas de Defesa**: Implementar contramedidas (differential privacy, etc.)
4. **Documentação**: Adicionar descobertas ao relatório final

## 🆘 Suporte

Para problemas ou dúvidas:
1. Verificar logs de erro no terminal
2. Consultar a documentação do código
3. Revisar os exemplos neste guia

---

**Nota**: Este experimento é para fins de pesquisa em segurança. Use responsavelmente.