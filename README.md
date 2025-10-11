# 🎓 Chatbot Abramede - Busca Semântica 100% Node.js

Sistema de chatbot que responde dúvidas de alunos **exclusivamente baseado no conteúdo do livro Abramede**, usando busca semântica com embeddings locais.

## 🎯 O que faz

- Aluno faz uma pergunta
- Sistema busca semanticamente no **texto completo do livro Abramede**
- Encontra as **páginas mais relevantes** usando embeddings
- Responde **APENAS com informações do livro** (não usa conhecimento geral da IA)
- Indica páginas específicas do livro + aulas e podcasts complementares

## 🚀 Vantagens

✅ **Tudo em Node.js** - sem precisar de Python!
✅ **Um único servidor** - não precisa rodar 2 serviços
✅ **Embeddings locais e grátis** - usando Transformers.js
✅ **Busca semântica inteligente** - entende contexto e sinônimos
✅ **Respostas confiáveis** - baseadas 100% no livro

## 📁 Estrutura do projeto

```
Chatbot/
├── server.js                      ← Servidor único (Node.js)
├── package.json
├── .env
├── gerar_embeddings_local.py     ← Script Python (roda 1x apenas para gerar embeddings)
├── public/
│   └── index.html
└── data/
    ├── abramede.pdf
    ├── abramede_texto.json       ← Texto extraído do PDF
    ├── embeddings_abramede.json  ← Embeddings gerados (1x)
    ├── indice_pesquisa_abramede.xlsx
    ├── Podcasts.xlsx
    └── Aulas.xlsx
```

## 🚀 Como instalar

### 1. Pré-requisitos

- **Node.js** 18+
- **Python** 3.8+ (apenas para gerar embeddings iniciais - 1 vez)
- Chave da API OpenAI

### 2. Instalar dependências Node.js

```bash
npm install
```

### 3. Configurar .env

Crie `.env` na raiz:

```env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
PORT=3000
```

### 4. Preparar os dados

Coloque os arquivos na pasta `data/`:

```bash
mkdir data
# Copie para data/:
# - abramede.pdf
# - indice_pesquisa_abramede.xlsx
# - Podcasts.xlsx
# - Aulas.xlsx
```

### 5. Gerar embeddings do livro (apenas 1 vez)

⚠️ **IMPORTANTE**: Execute ANTES de rodar o servidor.

```bash
# Instalar dependências Python
pip install PyPDF2 sentence-transformers tqdm

# Gerar embeddings
python gerar_embeddings_local.py
```

**O que faz:**
1. Extrai texto do PDF página por página
2. Gera embeddings localmente (sem gastar créditos OpenAI)
3. Salva em `data/embeddings_abramede.json`
4. **Tempo**: ~5-10 min para livro de 500 páginas
5. **Custo**: Grátis!

### 6. Rodar o servidor

```bash
npm start
```

**Na primeira execução:**
- O modelo de embeddings (~80MB) será baixado automaticamente
- Demora ~1 minuto (apenas na primeira vez)
- Depois inicia instantaneamente

Acesse: **http://localhost:3000**

## 💡 Como funciona

### Fluxo de busca:

1. **Aluno faz pergunta**: "Quanto é o alvo de ETCO2 na Ressuscitação?"

2. **Expande pergunta com OpenAI**: Identifica termos relacionados

3. **Gera embedding da pergunta**: Transformers.js (local, no Node.js)

4. **Busca por similaridade**: Compara com embeddings do livro

5. **Encontra páginas relevantes**: Top 5 páginas mais similares

6. **Monta prompt**: Envia para GPT apenas com contexto do livro

7. **Responde**: GPT responde baseado APENAS no material fornecido

### Exemplo de resposta:

```
O alvo de ETCO2 na ressuscitação é entre 35-45 mmHg. 
Valores abaixo de 10 mmHg indicam baixa eficácia nas compressões.

📖 PÁGINAS DO LIVRO CITADAS:
- Página 127: Monitorização durante RCP e uso de ETCO2
- Página 128: Metas de ETCO2 e interpretação de valores

📚 MATERIAIS COMPLEMENTARES:
- A1 (Aula): Reanimação ministrada por Dr. Denis Colares
- P1 (Podcast): Episódio 1 - REVISÃO ILCOR 2024 com Dr. Denis Colares
```

## 💰 Custos

### Setup inicial (1x apenas):
- Gerar embeddings: **Grátis** (local com Python)
- Download modelo Node.js: **Grátis** (~80MB)

### Por consulta de aluno:
- Expandir pergunta (OpenAI): **~$0.0001**
- Embedding da pergunta (Transformers.js): **Grátis** (local)
- Resposta GPT-4o-mini: **~$0.002**
- **Total: ~$0.002 por pergunta**

Com $5 de crédito = ~2.500 perguntas!

## 🔧 Personalização

### Número de páginas retornadas

No `server.js`, linha 101:

```javascript
const topPaginas = resultados.slice(0, 5); // Mudar de 5 para 3 ou 10
```

### Tamanho do contexto por página

Linha 105:

```javascript
const texto = pagina.texto.slice(0, 1500); // Aumentar para 2000 ou 3000
```

**Atenção**: Mais texto = mais tokens = mais caro

### Temperatura da resposta

Linha 170:

```javascript
temperature: 0.2, // 0.0 = mais fiel ao texto, 1.0 = mais criativo
```

## 🛠️ Regenerar embeddings

Se atualizar o PDF do livro:

```bash
# Deletar arquivos antigos
rm data/abramede_texto.json
rm data/embeddings_abramede.json

# Gerar novamente
python gerar_embeddings_local.py
```

## ✅ Vantagens dessa abordagem

✅ **Respostas confiáveis**: Baseadas 100% no livro
✅ **Busca semântica**: Entende contexto, não apenas palavras-chave
✅ **Cita fontes**: Sempre indica páginas específicas
✅ **Econômico**: Embeddings gerados localmente (custo único)
✅ **Rápido**: Busca em memória é instantânea
✅ **Escalável**: Funciona com livros de qualquer tamanho

## 🆘 Troubleshooting

### "embeddings_abramede.json não encontrado"

**Solução**: Execute `python gerar_embeddings_local.py`

### "PyPDF2 module not found"

**Solução**: `pip install PyPDF2`

### Respostas muito genéricas

**Solução**: Aumente número de páginas retornadas (topPaginas.slice(0, 10))

### Muito lento

**Problema**: Gerar embedding da pergunta demora 1-2s (normal)
**Não há como acelerar** - é o tempo da API OpenAI

## 📊 Dados dos arquivos e Sistema de IDs

O sistema organiza os materiais com IDs únicos para facilitar a referência:

### 🎓 Aulas (IDs: A1-A70)
- **A1, A2, A3...**: 70 aulas sobre diversos temas médicos
- Cada aula contém descrição detalhada dos tópicos abordados
- Exemplo: **A1** = Aula de Reanimação (ACLS, PALS, PCR)

### 🎙️ Podcasts (IDs: P1-P64)
- **P1, P2, P3...**: 64 episódios de podcast
- Cada episódio com tema específico
- Exemplo: **P1** = Episódio 1 - REVISÃO ILCOR 2024

### 📖 Índice Abramede (IDs: I1-I844)
- **I1, I2, I3...**: 844 entradas do índice remissivo do livro Abramede
- Referências organizadas por tema, subtema e seção
- Exemplo: **I20** = Referência sobre morte encefálica

### Como funciona na prática:

Quando o aluno pergunta algo, o chatbot responde indicando os IDs relevantes:

```
📚 ONDE ESTUDAR:
- A1 (Aula): Reanimação - cobre ACLS, PALS e uso de ETCO2
- P1 (Podcast): Episódio 1 revisa diretrizes ILCOR 2024
- I20 (Índice Abramede): Referência no livro sobre o tema
```

## 🔧 Personalização

### Ajustar número de itens do índice enviados

Por padrão, envia os primeiros 100 itens do índice Abramede como amostra. Para enviar mais ou menos:

No `server.js`, linha com `slice(0, 100)`:

```javascript
${indiceDados.indice.slice(0, 200).map(i => `${i.id}. ${i.ref}`).join('\n')}
... e mais ${indiceDados.indice.length - 200} entradas sobre diversos temas médicos
```

**Nota**: Enviar todos os 844 itens aumenta ~2k tokens por consulta (ainda muito barato)

### Mudar modelo OpenAI

No `server.js`:

```javascript
model: "gpt-4o-mini", // Opções: "gpt-4o" (mais inteligente, mais caro) ou "gpt-3.5-turbo" (mais barato)
```

### Ajustar temperatura (criatividade)

```javascript
temperature: 0.3, // 0.0 = mais preciso/determinístico, 1.0 = mais criativo
```

### Ajustar limite de tokens da resposta

```javascript
max_tokens: 500 // Aumentar para respostas mais longas
```

## 🛠️ Melhorias futuras possíveis

- [ ] Cache de perguntas frequentes (economia adicional)
- [ ] Sistema de embeddings para busca semântica
- [ ] Histórico de conversas
- [ ] Suporte a upload de novos arquivos
- [ ] Analytics de uso de tokens
- [ ] Versão em Docker

## 📝 Licença

MIT