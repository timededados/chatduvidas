// Script de teste para verificar instalação
const XLSX = require('xlsx');
const fs = require('fs');

console.log('🧪 Testando instalação do chatbot...\n');

// Testes
let erros = 0;

// 1. Verificar arquivos Excel
console.log('📁 Verificando arquivos Excel...');
const arquivos = [
  'indice_pesquisa_abramede.xlsx',
  'Podcasts.xlsx',
  'Aulas.xlsx'
];

arquivos.forEach(arquivo => {
  if (fs.existsSync(arquivo)) {
    console.log(`  ✅ ${arquivo}`);
    
    // Testar leitura
    try {
      const wb = XLSX.readFile(arquivo);
      const sheet = wb.Sheets[wb.SheetNames[0]];
      const data = XLSX.utils.sheet_to_json(sheet);
      console.log(`     └─ ${data.length} registros`);
    } catch (e) {
      console.log(`  ❌ Erro ao ler ${arquivo}: ${e.message}`);
      erros++;
    }
  } else {
    console.log(`  ❌ ${arquivo} não encontrado`);
    erros++;
  }
});

// 2. Verificar .env
console.log('\n🔑 Verificando configuração...');
if (fs.existsSync('.env')) {
  console.log('  ✅ Arquivo .env existe');
  
  // Ler e verificar API key
  const env = fs.readFileSync('.env', 'utf8');
  if (env.includes('OPENAI_API_KEY=sk-')) {
    console.log('  ✅ OPENAI_API_KEY configurada');
  } else {
    console.log('  ⚠️  OPENAI_API_KEY parece inválida');
  }
} else {
  console.log('  ❌ Arquivo .env não encontrado');
  console.log('     Crie um arquivo .env com: OPENAI_API_KEY=sua-chave');
  erros++;
}

// 3. Verificar pasta public
console.log('\n📄 Verificando frontend...');
if (fs.existsSync('public')) {
  console.log('  ✅ Pasta public existe');
  
  if (fs.existsSync('public/index.html')) {
    console.log('  ✅ index.html existe');
  } else {
    console.log('  ❌ index.html não encontrado em public/');
    erros++;
  }
} else {
  console.log('  ❌ Pasta public não existe');
  console.log('     Crie: mkdir public');
  erros++;
}

// 4. Verificar node_modules
console.log('\n📦 Verificando dependências...');
if (fs.existsSync('node_modules')) {
  console.log('  ✅ node_modules existe');
  
  const deps = ['express', 'cors', 'xlsx', 'openai'];
  deps.forEach(dep => {
    if (fs.existsSync(`node_modules/${dep}`)) {
      console.log(`  ✅ ${dep} instalado`);
    } else {
      console.log(`  ❌ ${dep} não instalado`);
      erros++;
    }
  });
} else {
  console.log('  ❌ node_modules não existe');
  console.log('     Execute: npm install');
  erros++;
}

// Resultado final
console.log('\n' + '='.repeat(50));
if (erros === 0) {
  console.log('✅ Tudo OK! Pode rodar: npm start');
} else {
  console.log(`❌ ${erros} erro(s) encontrado(s)`);
  console.log('   Corrija os erros acima antes de rodar');
}
console.log('='.repeat(50) + '\n');