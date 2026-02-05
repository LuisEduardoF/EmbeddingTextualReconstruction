# Análise de Riscos de Reconstrução Textual em Modelos de Predição de Conciliação Trabalhista

## 📋 Visão Geral

Este projeto investiga vulnerabilidades de segurança em sistemas de Inteligência Artificial aplicados ao Judiciário brasileiro, especificamente analisando riscos de **inversão de embeddings** em modelos de predição de conciliações trabalhistas baseados em BERT.

## 🎯 Problema e Motivação

O Judiciário brasileiro tem adotado técnicas de Inteligência Artificial e Processamento de Linguagem Natural (PLN), utilizando modelos como BERTimbau para análise de processos trabalhistas. Embora as representações vetoriais (embeddings) sejam frequentemente consideradas "seguras" por serem numéricas, este trabalho investiga se esses vetores retêm informações semânticas suficientes para permitir:

- **Reconstrução do texto original** a partir dos embeddings
- **Inferência de atributos sensíveis** através de ataques de inversão
- **Vazamento de informações confidenciais** na fase de inferência

## 🎓 Objetivos

### Objetivo Geral

Avaliar experimentalmente a segurança dos embeddings gerados pelo modelo BERTimbau fine-tuned, verificando a viabilidade de recuperar informações textuais de processos judiciais a partir apenas de suas representações vetoriais.

### Produtos Esperados

1. **Modelo Atacante**: Decodificador neural treinado para reverter embeddings em texto
2. **Relatório de Auditoria**: Quantificação da taxa de sucesso na recuperação de tokens e palavras-chave sensíveis
3. **Análise de Risco**: Avaliação sobre a suficiência da anonimização prévia e correlações semânticas perigosas

## 🔬 Metodologia

### Tipo de Trabalho

**Desenvolvimento Experimental / Protótipo**

Este trabalho aproveita o pipeline técnico já desenvolvido na minha Tese de Mestrado, adicionando uma camada adversária para validação prática dos riscos de segurança.

### Pipeline de Ataque

```
Texto Original → BERTimbau → Embedding [CLS] → Modelo Atacante → Texto Reconstruído
                  (Fine-tuned)                    (Inversor)
```

## 📊 Atividades do Projeto

### 1. Geração de Embeddings
- Utilização do pipeline de pré-processamento existente
- Extração de embeddings do token `[CLS]` usando BERTimbau fine-tuned
- Criação de dataset intermediário: `(Vetor Embedding) → (Texto Original)`

### 2. Desenvolvimento do Modelo Adversário
- Implementação de rede neural "atacante"
- Arquitetura projetada para reconstrução textual a partir de embeddings
- Predição de termos sensíveis (ex: nomes de litigantes, doenças ocupacionais)

### 3. Execução do Ataque de Inversão
- Treinamento do modelo adversário
- Uso de divisão temporal consistente com o projeto original
- Simulação de vazamento de dados em condições realistas

### 4. Análise de Vulnerabilidade
- Comparação entre texto reconstruído e original
- Cálculo de métricas de vazamento de informação
- Avaliação quantitativa da eficácia do ataque

### 5. Consolidação e Documentação
- Relatório final conectando resultados aos conceitos de confidencialidade e privacidade em IA
- Recomendações de segurança para sistemas judiciais

## 📅 Cronograma

| Checkpoint | Data | Entregável |
|------------|------|------------|
| **Checkpoint 1** | 21/11 | Apresentação do Conceito + Revisão da Literatura |
| **Checkpoint 2** | 18/12 | Definição da Arquitetura do Modelo Atacante + Resultados Parciais |
| **Checkpoint 3** | 05/02 | Código Completo + Relatório Final + Resultados |

## 🛠️ Tecnologias

- **Modelo Base**: BERTimbau
- **Framework**: PyTorch / TensorFlow
- **Linguagem**: Python
- **Domínio**: Processamento de Linguagem Natural (PLN)
- **Área**: Segurança em Machine Learning

## 📖 Estrutura do Repositório

```
.
├── README.md                 # Este arquivo
├── data/                     # Dados e embeddings
├── models/                   # Modelos treinados
│   ├── bertimbau/           # Modelo BERTimbau fine-tuned
│   └── attacker/            # Modelo adversário (inversor)
├── src/                      # Código fonte
│   ├── preprocessing/       # Pipeline de pré-processamento
│   ├── embedding/           # Geração de embeddings
│   ├── attack/              # Implementação do ataque
│   └── evaluation/          # Métricas e análises
├── notebooks/               # Jupyter notebooks para análise
├── results/                 # Resultados e visualizações
└── docs/                    # Documentação adicional
```

## 📄 Licença

MIT License

## 👥 Autores

Luís Câmara - Mestrando PPGI UFES

## 📧 Contato

luisefcamara@gmail.com | [GitHub](https://github.com/LuisEduardoF) (git) | [Linkedin](https://www.linkedin.com/in/luisefcamara/)
---