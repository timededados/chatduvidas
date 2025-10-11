# 🔧 Troubleshooting

## Problemas comuns e soluções

### ❌ Erro: "Cannot find module 'express'"

**Problema**: Dependências não instaladas

**Solução**:
```bash
npm install
```

---

### ❌ Erro: "ENOENT: no such file or directory"

**Problema**: Arquivos Excel não encontrados

**Solução**:
- Verifique se os 3 arquivos Excel estão na raiz do projeto
- Nomes devem ser exatamente:
  - `indice_pesquisa_abramede.xlsx`
  - `Podcasts.xlsx`
  - `Aulas.xlsx`

---

### ❌ Erro: "Invalid API key"

**Problema**: Chave da OpenAI inválida ou não configurada

**Solução**:
1. Crie arquivo `.env` na raiz do projeto
2. Adicione: `OPENAI_API_KEY=sua-chave-aqui`
3. Obtenha chave em: https://platform.openai.com/api-keys

---

### ❌ Erro: "CORS blocked"

**Problema**: Frontend não consegue conectar ao backend

**Solução**:
- Verifique se o servidor está rodando (`npm start`)
- Acesse pelo endereço correto: `http://localhost:3000`
- Não use `file://` no navegador

---

### ❌ Página em branco

**Problema**: Arquivo HTML não encontrado

**Solução**:
1. Crie pasta `public`: `mkdir public`
2. Coloque `index.html` dentro de `public/`
3. Reinicie o servidor

---

### ⚠️ Respostas muito genéricas

**Problema**: Poucos resultados sendo enviados para OpenAI

**Solução**:
No `server.js`, aumente o limite de resultados:
```javascript
resultados.indice.splice(10); // Era 5, agora 10
resultados.podcasts.splice(5); // Era 3, agora 5
resultados.aulas.splice(5); // Era 3, agora 5
```

**Nota**: Isso aumentará o consumo de tokens

---

### 💰 Gastando muitos tokens?

**Problema**: Consumo de API muito alto

**Soluções**:

1. **Reduza resultados enviados**:
```javascript
resultados.indice.splice(3); // Reduzir de 5 para 3
```

2. **Reduza max_tokens**:
```javascript
max_tokens: 200 // Era 300
```

3. **Use modelo mais barato**:
```javascript
model: "gpt-4o-mini" // Já é o mais barato, mantém assim
```

---

### 🐌 Respostas muito lentas

**Problema**: Demora para responder

**Possíveis causas e soluções**:

1. **Conexão com OpenAI**:
   - Normal: 2-5 segundos
   - Se > 10 segundos, verifique internet

2. **Arquivos muito grandes**:
   - Se arquivos > 5MB, busca local pode demorar
   - Solução: Otimizar busca ou criar índice em memória

3. **Muitos resultados**:
   - Reduza `.splice()` para números menores

---

### 🔍 Não encontra nada

**Problema**: Sempre retorna "Não encontrei conteúdo"

**Soluções**:

1. **Teste busca local**:
```javascript
// Adicione console.log no server.js
console.log('Resultados:', resultados);
```

2. **Verifique acentuação**:
   - Sistema busca sem acentos automaticamente
   - Teste com termos simples: "febre", "lombalgia"

3. **Termos muito específicos**:
   - Use termos mais gerais
   - Exemplo: ao invés de "terapia trombolítica", tente "trombólise"

---

### 🔄 Mudanças não aparecem

**Problema**: Alterou código mas não funciona

**Solução**:
1. Reinicie o servidor (Ctrl+C e `npm start`)
2. Limpe cache do navegador (Ctrl+Shift+R)
3. Se usar `npm run dev`, deveria reiniciar automaticamente

---

### 📊 Como ver uso de tokens

**Solução**: Os tokens são exibidos no frontend após cada resposta

Para ver no console do servidor, adicione no `server.js`:
```javascript
console.log('Tokens usados:', completion.usage.total_tokens);
```

---

## 🆘 Precisa de mais ajuda?

1. Verifique os logs do servidor no terminal
2. Abra DevTools do navegador (F12) e veja Console
3. Teste com perguntas simples primeiro: "lombalgia", "febre"