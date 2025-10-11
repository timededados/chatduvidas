// =======================================================
// CHATBOT ABRAMEDE – Busca semântica com múltiplas variações
// Busca com pergunta original + variações, valida com OpenAI
// =======================================================

import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import XLSX from 'xlsx';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import OpenAI from 'openai';
import { pipeline } from '@xenova/transformers';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY
});

const DATA_DIR = path.join(__dirname, 'data');

// Modelo de embeddings (será carregado na inicialização)
let extractor = null;

let dados = {
  indice: [],
  podcasts: [],
  aulas: [],
  paginasLivro: [],
  embeddingsLivro: [],
  sumario: '' // Sumário do livro (páginas 46-59)
};

function carregarDados() {
  const arquivosNecessarios = [
    'indice_pesquisa_abramede.xlsx',
    'Podcasts.xlsx',
    'Aulas.xlsx',
    'abramede_texto.json',
    'embeddings_abramede.json'
  ];

  const arquivosFaltando = arquivosNecessarios.filter(arquivo => 
    !fs.existsSync(path.join(DATA_DIR, arquivo))
  );

  if (arquivosFaltando.length > 0) {
    console.error('\n❌ ERRO: Arquivos não encontrados:\n');
    arquivosFaltando.forEach(arquivo => console.error(`   ❌ ${arquivo}`));
    process.exit(1);
  }

  const wbIndice = XLSX.readFile(path.join(DATA_DIR, 'indice_pesquisa_abramede.xlsx'));
  dados.indice = XLSX.utils.sheet_to_json(wbIndice.Sheets[wbIndice.SheetNames[0]]);

  const wbPodcasts = XLSX.readFile(path.join(DATA_DIR, 'Podcasts.xlsx'));
  dados.podcasts = XLSX.utils.sheet_to_json(wbPodcasts.Sheets[wbPodcasts.SheetNames[0]]);

  const wbAulas = XLSX.readFile(path.join(DATA_DIR, 'Aulas.xlsx'));
  dados.aulas = XLSX.utils.sheet_to_json(wbAulas.Sheets[wbAulas.SheetNames[0]]);

  dados.paginasLivro = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'abramede_texto.json'), 'utf-8'));
  dados.embeddingsLivro = JSON.parse(fs.readFileSync(path.join(DATA_DIR, 'embeddings_abramede.json'), 'utf-8'));

  // Extrair sumário (páginas 46-59)
  const paginasSumario = dados.paginasLivro.filter(p => p.pagina >= 46 && p.pagina <= 59);
  dados.sumario = paginasSumario.map(p => `[Página ${p.pagina}]\n${p.texto}`).join('\n\n');
  
  console.log('✅ Dados carregados:', {
    indice: dados.indice.length,
    podcasts: dados.podcasts.length,
    aulas: dados.aulas.length,
    paginasLivro: dados.paginasLivro.length,
    embeddingsLivro: dados.embeddingsLivro.length,
    sumario: dados.sumario.length > 0 ? `${paginasSumario.length} páginas` : '❌ não encontrado'
  });
  
  if (dados.aulas[0]) {
    console.log('📋 Primeira aula:', {
      nome: dados.aulas[0].Aula,
      autor: dados.aulas[0].Autor || '❌ Coluna Autor não encontrada'
    });
  }
  if (dados.podcasts[0]) {
    console.log('🎙️ Primeiro podcast:', {
      episodio: dados.podcasts[0].Episódio,
      autor: dados.podcasts[0].Autor || '❌ Coluna Autor não encontrada'
    });
  }
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function gerarEmbedding(texto) {
  if (!extractor) {
    throw new Error('Modelo de embeddings não carregado');
  }
  
  // Gerar embedding usando Transformers.js
  const output = await extractor(texto, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

app.post('/api/chat', async (req, res) => {
  try {
    const { pergunta } = req.body;

    if (!pergunta || pergunta.trim().length === 0) {
      return res.json({ resposta: 'Por favor, faça uma pergunta.', fontes: [] });
    }

    console.log(`\n🔍 Pergunta original: "${pergunta}"`);

    // 1️⃣ GERAR MÚLTIPLAS VARIAÇÕES da pergunta
    console.log('🧠 Gerando variações da pergunta...');
    let variacoes = [pergunta]; // Começa com a original
    
    try {
      const variacoesPrompt = `Você é especialista em medicina de emergência. Gere 3 VARIAÇÕES DIFERENTES da pergunta abaixo, usando sinônimos médicos e formas alternativas de perguntar a MESMA coisa.

Pergunta: "${pergunta}"

INSTRUÇÕES:
- Cada variação deve manter o MESMO significado
- Use sinônimos médicos (RCP = reanimação cardiopulmonar, PCR = parada, ETCO2 = capnografia)
- Use termos técnicos alternativos
- Mantenha perguntas completas e coerentes

Retorne EXATAMENTE 3 variações, uma por linha, SEM numeração, SEM explicações:`;

      const variacoesResponse = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: variacoesPrompt }],
        temperature: 0.5,
        max_tokens: 300
      });

      const variacoesTexto = variacoesResponse.choices[0].message.content.trim();
      const variacoesGeradas = variacoesTexto
        .split('\n')
        .map(v => v.replace(/^\d+\.\s*/, '').trim())
        .filter(v => v.length > 10);
      
      variacoes = [pergunta, ...variacoesGeradas];
      console.log(`✅ ${variacoes.length} versões geradas (original + variações):`);
      variacoes.forEach((v, i) => console.log(`   ${i + 1}. ${v}`));
      
    } catch (error) {
      console.error('⚠️  Erro ao gerar variações:', error.message);
      console.log('📌 Continuando apenas com pergunta original');
    }

    // 2️⃣ CONSULTAR SUMÁRIO para identificar capítulos relevantes
    console.log('\n📚 Consultando sumário para identificar capítulos...');
    let paginasAlvo = []; // Páginas dos capítulos identificados
    
    try {
      const sumarioPrompt = `Você é um especialista em medicina de emergência. Analise o sumário abaixo e identifique TODOS os capítulos/seções que podem conter informações relevantes para responder a pergunta.

PERGUNTA: "${pergunta}"

SUMÁRIO DO LIVRO ABRAMEDE:
${dados.sumario}

INSTRUÇÕES:
- Identifique TODOS os capítulos/seções relevantes (pode ser mais de um)
- Para cada capítulo, extraia o intervalo de páginas mencionado no sumário
- Se não houver páginas explícitas, estime baseado na estrutura
- Seja INCLUSIVO: inclua capítulos que possam ter relação mesmo que indireta

Retorne APENAS uma lista de intervalos de páginas no formato:
[página_inicial]-[página_final]
[página_inicial]-[página_final]

Exemplo:
125-145
230-250

Se não encontrar nenhum capítulo relevante, retorne: NENHUM`;

      const sumarioResponse = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: sumarioPrompt }],
        temperature: 0.3,
        max_tokens: 300
      });

      const resultado = sumarioResponse.choices[0].message.content.trim();
      
      if (resultado.toUpperCase() !== 'NENHUM') {
        // Extrair intervalos de páginas
        const intervalos = resultado.match(/(\d+)-(\d+)/g) || [];
        
        intervalos.forEach(intervalo => {
          const [inicio, fim] = intervalo.split('-').map(Number);
          for (let p = inicio; p <= fim; p++) {
            if (!paginasAlvo.includes(p)) {
              paginasAlvo.push(p);
            }
          }
        });
        
        console.log(`✅ Capítulos identificados: ${intervalos.length}`);
        console.log(`📄 Páginas alvo: ${paginasAlvo.length} páginas (${paginasAlvo[0]}-${paginasAlvo[paginasAlvo.length-1]})`);
        intervalos.forEach(int => console.log(`   Capítulo: páginas ${int}`));
      } else {
        console.log('⚠️  Nenhum capítulo específico identificado no sumário');
      }
      
    } catch (error) {
      console.error('⚠️  Erro ao consultar sumário:', error.message);
    }
    
    // Se não encontrou páginas específicas, buscar em todo o livro
    if (paginasAlvo.length === 0) {
      console.log('📌 Buscando em todo o livro (sem filtro de capítulo)');
      paginasAlvo = dados.paginasLivro.map(p => p.pagina);
    }

    // 3️⃣ BUSCAR apenas nas páginas dos capítulos identificados
    console.log('\n🔎 Buscando nos capítulos identificados...');
    const todasPaginas = new Map(); // pagina -> melhor score
    const aulasRelevantes = new Map(); // id aula -> {aula, score}
    const podcastsRelevantes = new Map(); // id podcast -> {podcast, score}
    
    // Filtrar embeddings apenas das páginas alvo
    const embeddingsFiltrados = dados.embeddingsLivro.filter(item => 
      paginasAlvo.includes(item.pagina)
    );
    
    console.log(`   Buscando em ${embeddingsFiltrados.length} páginas filtradas (de ${dados.embeddingsLivro.length} totais)`);
    
    for (let i = 0; i < variacoes.length; i++) {
      const variacao = variacoes[i];
      console.log(`   Variação ${i + 1}/${variacoes.length}...`);
      
      const embedding = await gerarEmbedding(variacao);
      
      // Buscar apenas nas páginas filtradas
      embeddingsFiltrados.forEach(item => {
        const score = cosineSimilarity(embedding, item.embedding);
        const scoreAtual = todasPaginas.get(item.pagina) || 0;
        
        if (score > scoreAtual) {
          todasPaginas.set(item.pagina, score);
        }
      });
      
      // Buscar aulas relevantes no índice
      dados.indice.forEach((item, idx) => {
        if (item.Tipo === 'Aula') {
          // Criar texto de busca com todos os campos relevantes
          const textoAula = `${item.Tema || ''} ${item.Descrição || ''} ${item.Tags || ''}`.toLowerCase();
          const textoVariacao = variacao.toLowerCase();
          
          // Busca por palavras-chave
          let score = 0;
          const palavrasVariacao = textoVariacao.split(/\s+/).filter(p => p.length > 3);
          palavrasVariacao.forEach(palavra => {
            if (textoAula.includes(palavra)) {
              score += 0.15; // Score por palavra encontrada
            }
          });
          
          if (score > 0) {
            const scoreAtual = aulasRelevantes.get(idx)?.score || 0;
            if (score > scoreAtual) {
              aulasRelevantes.set(idx, { aula: item, score });
            }
          }
        }
      });
      
      // Buscar podcasts relevantes no índice
      dados.indice.forEach((item, idx) => {
        if (item.Tipo === 'Podcast') {
          const textoPodcast = `${item.Tema || ''} ${item.Descrição || ''} ${item.Tags || ''}`.toLowerCase();
          const textoVariacao = variacao.toLowerCase();
          
          let score = 0;
          const palavrasVariacao = textoVariacao.split(/\s+/).filter(p => p.length > 3);
          palavrasVariacao.forEach(palavra => {
            if (textoPodcast.includes(palavra)) {
              score += 0.15;
            }
          });
          
          if (score > 0) {
            const scoreAtual = podcastsRelevantes.get(idx)?.score || 0;
            if (score > scoreAtual) {
              podcastsRelevantes.set(idx, { podcast: item, score });
            }
          }
        }
      });
    }
    
    // Ordenar páginas do livro
    const paginasOrdenadas = Array.from(todasPaginas.entries())
      .map(([pagina, score]) => ({ pagina, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);

    console.log(`\n✅ Resultados da busca:`);
    console.log(`   📖 ${paginasOrdenadas.length} páginas encontradas nos capítulos`);
    console.log(`   📚 ${aulasRelevantes.size} aulas relevantes`);
    console.log(`   🎙️  ${podcastsRelevantes.size} podcasts relevantes`);
    
    console.log('\n📄 Top 10 páginas dos capítulos:');
    paginasOrdenadas.forEach(p => {
      console.log(`   Página ${p.pagina}: ${(p.score * 100).toFixed(1)}%`);
    });
    
    if (aulasRelevantes.size > 0) {
      console.log('\n📚 Aulas encontradas:');
      Array.from(aulasRelevantes.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .forEach(({aula, score}) => {
          console.log(`   ${aula.Tema} (${(score * 100).toFixed(1)}%)`);
        });
    }
    
    if (podcastsRelevantes.size > 0) {
      console.log('\n🎙️  Podcasts encontrados:');
      Array.from(podcastsRelevantes.values())
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .forEach(({podcast, score}) => {
          console.log(`   ${podcast.Tema} (${(score * 100).toFixed(1)}%)`);
        });
    }

    // 4️⃣ VALIDAR se os capítulos contêm informação relevante
    console.log('\n🔍 Validando conteúdo dos capítulos com OpenAI...');
    
    // Pegar texto de uma amostra das páginas mais relevantes
    const amostraCapitulos = paginasOrdenadas.slice(0, 5).map(p => {
      const paginaObj = dados.paginasLivro.find(pg => pg.pagina === p.pagina);
      return paginaObj ? `[Página ${p.pagina}] ${paginaObj.texto.slice(0, 800)}` : '';
    }).filter(Boolean).join('\n\n');

    let capitulosTemConteudo = false;
    
    try {
      const validacaoCapitulosPrompt = `Analise se o conteúdo abaixo dos capítulos identificados REALMENTE contém informações que podem responder à pergunta, mesmo que parcialmente.

PERGUNTA: "${pergunta}"

AMOSTRA DO CONTEÚDO DOS CAPÍTULOS:
${amostraCapitulos}

Responda APENAS:
- SIM: se há informação específica relacionada à pergunta
- NÃO: se os capítulos não abordam o tema da pergunta

Resposta (SIM ou NÃO):`;

      const validacaoCapitulos = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [{ role: 'user', content: validacaoCapitulosPrompt }],
        temperature: 0,
        max_tokens: 10
      });

      const resultado = validacaoCapitulos.choices[0].message.content.trim().toUpperCase();
      capitulosTemConteudo = resultado.includes('SIM');
      
      if (capitulosTemConteudo) {
        console.log('✅ Capítulos contêm informação relevante');
      } else {
        console.log('❌ Capítulos não contêm informação sobre o tema');
      }
      
    } catch (error) {
      console.error('⚠️  Erro ao validar capítulos:', error.message);
      capitulosTemConteudo = true; // Em caso de erro, continua com a busca
    }
    
    if (!capitulosTemConteudo) {
      return res.json({
        resposta: '❌ Desculpe, não encontrei informações no livro Abramede que respondam à sua pergunta. O sumário foi consultado e os capítulos relacionados não contêm o conteúdo necessário.',
        fontes: { paginas: [], aulas: [], podcasts: [] }
      });
    }

    // 5️⃣ VALIDAR páginas específicas
    console.log('\n🔍 Validando páginas individuais com OpenAI...');
    const paginasValidadas = [];
    
    for (const candidata of paginasOrdenadas) {
      const paginaObj = dados.paginasLivro.find(p => p.pagina === candidata.pagina);
      if (!paginaObj) continue;
      
      const textoCompleto = paginaObj.texto;
      
      try {
        const validacaoPrompt = `Você é um validador de relevância. Analise se o texto abaixo REALMENTE contém informações que respondem à pergunta.

PERGUNTA: "${pergunta}"

TEXTO (Página ${candidata.pagina}):
${textoCompleto.slice(0, 2000)}

Responda APENAS:
- SIM: se o texto contém informação específica que responde a pergunta
- NÃO: se o texto não responde ou só menciona termos relacionados sem responder

Resposta (SIM ou NÃO):`;

        const validacao = await openai.chat.completions.create({
          model: 'gpt-4o-mini',
          messages: [{ role: 'user', content: validacaoPrompt }],
          temperature: 0,
          max_tokens: 10
        });

        const resultado = validacao.choices[0].message.content.trim().toUpperCase();
        
        if (resultado.includes('SIM')) {
          paginasValidadas.push({
            pagina: candidata.pagina,
            texto: textoCompleto,
            score: candidata.score
          });
          console.log(`   ✅ Página ${candidata.pagina}: VALIDADA`);
          
          // Parar após encontrar 5 páginas validadas
          if (paginasValidadas.length >= 5) break;
        } else {
          console.log(`   ❌ Página ${candidata.pagina}: rejeitada (não responde)`);
        }
        
      } catch (error) {
        console.error(`   ⚠️  Erro validando página ${candidata.pagina}:`, error.message);
      }
    }
    
    console.log(`\n✅ ${paginasValidadas.length} páginas validadas`);
    
    // Se nenhuma página foi validada, retornar mensagem apropriada
    if (paginasValidadas.length === 0) {
      return res.json({
        resposta: '❌ Desculpe, não encontrei informações no livro Abramede que respondam especificamente à sua pergunta. Por favor, reformule ou faça uma pergunta diferente.',
        fontes: { paginas: [], aulas: [], podcasts: [] }
      });
    }
    
    // Preparar texto das páginas validadas
    const textoPaginas = paginasValidadas.map(p => {
      return `📖 Página ${p.pagina} (${(p.score * 100).toFixed(1)}% relevância):\n${p.texto.slice(0, 1500)}\n`;
    }).join('\n---\n\n');

    // 6️⃣ PREPARAR materiais complementares
    // Criar índice apenas com materiais relevantes encontrados
    const aulasEncontradas = Array.from(aulasRelevantes.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 10) // Top 10 aulas
      .map(({aula}) => aula);
    
    const podcastsEncontrados = Array.from(podcastsRelevantes.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, 10) // Top 10 podcasts
      .map(({podcast}) => podcast);
    
    const indiceDados = {
      aulas: aulasEncontradas.map((a, i) => {
        // Encontrar na lista completa para pegar autor
        const aulaCompleta = dados.aulas.find(ac => 
          ac.Aula === a.Tema || ac.Aula?.includes(a.Tema)
        );
        return {
          id: `A${i + 1}`,
          nome: a.Tema,
          autor: aulaCompleta?.Autor || 'Não informado'
        };
      }),
      podcasts: podcastsEncontrados.map((p, i) => {
        // Encontrar na lista completa para pegar autor
        const podcastCompleto = dados.podcasts.find(pc => 
          pc.Episódio === p.Tema || pc.Tema === p.Tema
        );
        return {
          id: `P${i + 1}`,
          episodio: podcastCompleto?.Episódio || p.Tema,
          tema: p.Tema,
          autor: podcastCompleto?.Autor || 'Não informado'
        };
      })
    };

    // 7️⃣ MONTAR RESPOSTA (apenas com conteúdo validado)
    let materiaisComplementares = '';
    if (indiceDados.aulas.length > 0 || indiceDados.podcasts.length > 0) {
      materiaisComplementares = '\nMATERIAIS COMPLEMENTARES RELEVANTES:\n';
      if (indiceDados.aulas.length > 0) {
        materiaisComplementares += `AULAS: ${indiceDados.aulas.map(a => `${a.id}. ${a.nome}`).join(', ')}\n`;
      }
      if (indiceDados.podcasts.length > 0) {
        materiaisComplementares += `PODCASTS: ${indiceDados.podcasts.map(p => `${p.id}. ${p.episodio} - ${p.tema}`).join(', ')}`;
      }
    }
    
    const prompt = `Você é um assistente educacional especializado em medicina de emergência ABRAMEDE.

PERGUNTA: "${pergunta}"

CONTEÚDO VALIDADO DO LIVRO ABRAMEDE:
${textoPaginas}
${materiaisComplementares}

⚠️ REGRAS OBRIGATÓRIAS:
1. Use APENAS o conteúdo do livro acima - NUNCA use conhecimento geral
2. SEMPRE cite as páginas específicas (ex: "Conforme página 45...")
3. Se o conteúdo não responder completamente, diga "O livro não detalha..."
4. NUNCA invente informações
5. Se houver materiais complementares acima, indique-os quando relevantes usando os IDs (A1, P1, etc)

FORMATO OBRIGATÓRIO:
[Resposta baseada no livro, citando páginas específicas]

📖 PÁGINAS CITADAS:
- Página [X]: [resumo do que foi usado]
- Página [Y]: [resumo do que foi usado]

📚 MATERIAIS COMPLEMENTARES (se houver materiais relevantes acima):
- A[X] (Aula): [nome]
- P[X] (Podcast): [episódio] - [tema]`;

    // Gerar resposta
    console.log('\n🤖 Gerando resposta baseada em conteúdo validado...');
    const completion = await openai.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.2,
      max_tokens: 800
    });

    let resposta = completion.choices[0].message.content;

    // 8️⃣ ENRIQUECER com autores
    const idsAulas = [...new Set(resposta.match(/A\d+/g) || [])];
    const idsPodcasts = [...new Set(resposta.match(/P\d+/g) || [])];

    if (idsAulas.length > 0 || idsPodcasts.length > 0) {
      console.log('📝 Enriquecendo com autores...');
    }
    
    idsAulas.forEach(id => {
      const num = parseInt(id.substring(1)) - 1;
      const aula = indiceDados.aulas[num];
      if (aula && aula.autor !== 'Não informado') {
        console.log(`   ${id}: ${aula.nome} - ${aula.autor}`);
        const pattern = `${id} (Aula): ${aula.nome}`;
        const replacement = `${id} (Aula): ${aula.nome} ministrada por ${aula.autor}`;
        resposta = resposta.replace(pattern, replacement);
      }
    });
    
    idsPodcasts.forEach(id => {
      const num = parseInt(id.substring(1)) - 1;
      const podcast = indiceDados.podcasts[num];
      if (podcast && podcast.autor !== 'Não informado') {
        console.log(`   ${id}: ${podcast.episodio} - ${podcast.tema} - ${podcast.autor}`);
        
        const lines = resposta.split('\n').map(line => {
          if (line.includes(`${id} (Podcast)`) && !line.includes(' com ')) {
            if (line.includes(podcast.tema)) {
              return line + ` com ${podcast.autor}`;
            } else if (line.includes(podcast.episodio)) {
              return line.replace(
                podcast.episodio,
                `${podcast.episodio} - ${podcast.tema}`
              ) + ` com ${podcast.autor}`;
            }
          }
          return line;
        });
        resposta = lines.join('\n');
      }
    });

    const fontes = {
      paginas: paginasValidadas.map(p => ({ pagina: p.pagina, score: p.score })),
      aulas: idsAulas.map(id => {
        const num = parseInt(id.substring(1)) - 1;
        return indiceDados.aulas[num];
      }).filter(Boolean),
      podcasts: idsPodcasts.map(id => {
        const num = parseInt(id.substring(1)) - 1;
        return indiceDados.podcasts[num];
      }).filter(Boolean)
    };

    console.log('✅ Resposta gerada com sucesso');
    console.log(`📚 Sumário consultado: capítulos identificados`);
    console.log(`📄 ${fontes.paginas.length} páginas validadas e citadas`);
    console.log(`📚 ${fontes.aulas.length} aulas + ${fontes.podcasts.length} podcasts indicados`);
    if (fontes.aulas.length === 0 && aulasRelevantes.size > 0) {
      console.log('⚠️  Aulas encontradas mas não incluídas na resposta (GPT não considerou relevante)');
    }
    if (fontes.podcasts.length === 0 && podcastsRelevantes.size > 0) {
      console.log('⚠️  Podcasts encontrados mas não incluídos na resposta (GPT não considerou relevante)');
    }
    console.log('');

    res.json({
      resposta,
      fontes,
      tokens: completion.usage.total_tokens
    });

  } catch (error) {
    console.error('❌ Erro:', error);
    res.status(500).json({
      erro: 'Erro ao processar pergunta',
      detalhes: error.message
    });
  }
});

app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    modeloCarregado: extractor !== null,
    dados: {
      indice: dados.indice.length,
      podcasts: dados.podcasts.length,
      aulas: dados.aulas.length,
      paginasLivro: dados.paginasLivro.length,
      embeddingsLivro: dados.embeddingsLivro.length,
      sumario: dados.sumario.length > 0
    }
  });
});

async function inicializarModelo() {
  console.log('🔧 Carregando modelo de embeddings...');
  console.log('   Modelo: Xenova/all-MiniLM-L6-v2');
  console.log('   (Primeira vez pode demorar ~1 minuto para baixar)\n');
  
  try {
    extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    console.log('✅ Modelo carregado com sucesso!\n');
    return true;
  } catch (error) {
    console.error('❌ Erro ao carregar modelo:', error.message);
    return false;
  }
}

const PORT = process.env.PORT || 3000;

async function iniciar() {
  console.log('🚀 INICIANDO CHATBOT ABRAMEDE\n');
  console.log('='.repeat(60));
  
  // 1. Carregar dados
  carregarDados();
  
  // 2. Carregar modelo de embeddings
  const modeloOk = await inicializarModelo();
  
  if (!modeloOk) {
    console.error('❌ Não foi possível carregar o modelo de embeddings');
    console.error('   O chatbot não funcionará corretamente');
  }
  
  // 3. Iniciar servidor
  app.listen(PORT, () => {
    console.log('='.repeat(60));
    console.log(`✅ Servidor rodando na porta ${PORT}`);
    console.log(`📚 Sumário: ${dados.sumario.length > 0 ? 'Carregado (páginas 46-59)' : '❌ Não encontrado'}`);
    console.log(`📊 Livro: ${dados.paginasLivro.length} páginas indexadas`);
    console.log(`📚 Materiais: ${dados.aulas.length} aulas, ${dados.podcasts.length} podcasts`);
    console.log(`🔧 Modelo: ${modeloOk ? 'Carregado' : '❌ Erro'}`);
    console.log('\n🌐 Acesse: http://localhost:' + PORT);
    console.log('='.repeat(60) + '\n');
  });
}

iniciar();