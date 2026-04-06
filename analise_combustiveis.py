# TRABALHO 03 - ESTATÍSTICA
# Teste de Hipótese e Teste t aplicado a dados públicos de preços de combustíveis
# 
# Este script realiza:
# 1. Download e carregamento da base da ANP
# 2. Filtro por 2 cidades (pelo menos 1 do Piauí), 1 combustível e mínimo 30 semanas
# 3. Montagem da base pareada por semana
# 4. Estatísticas descritivas
# 5. Teste t pareado
# 6. Gráficos: histogramas e boxplot
# 7. Relatório final

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
import os
import warnings
warnings.filterwarnings('ignore')

# Configurações
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# -------------------------------------------------------------------
# 1. Carregar a base da ANP
# -------------------------------------------------------------------
arquivo_csv = "serie_historica_precos.csv"

if not os.path.exists(arquivo_csv):
    print(f"Arquivo {arquivo_csv} não encontrado!")
    print("Coloque o arquivo CSV na mesma pasta deste script.")
    exit()

print("Carregando base de dados...")
df = pd.read_csv(arquivo_csv, sep=';', encoding='latin1', low_memory=False)
print(f"Base carregada. Linhas totais: {len(df):,}")

# -------------------------------------------------------------------
# 2. Configurações do recorte (VOCÊ PODE ALTERAR AQUI)
# -------------------------------------------------------------------
# Combustível (exemplos: 'GASOLINA COMUM', 'ETANOL', 'DIESEL')
combustivel = 'GASOLINA'

# Cidade do Piauí (obrigatória)
cidade_piaui = 'TERESINA'

# Outra cidade (pode ser de qualquer estado)
cidade_outra = 'FORTALEZA'

print(f"\nConfiguração:")
print(f"  Combustível: {combustivel}")
print(f"  Cidade 1 (PI): {cidade_piaui}")
print(f"  Cidade 2: {cidade_outra}")

# -------------------------------------------------------------------
# 3. Verificar dados disponíveis
# -------------------------------------------------------------------
# Mostrar produtos disponíveis
print(f"\nProdutos disponíveis na base:")
produtos = df['Produto'].value_counts().head(10)
for prod, qtd in produtos.items():
    print(f"  - {prod}: {qtd:,} registros")

# Mostrar cidades do Piauí
print(f"\nCidades do Piauí com dados:")
cidades_pi = df[df['Estado - Sigla'] == 'PI']['Municipio'].value_counts().head(10)
for cidade, qtd in cidades_pi.items():
    print(f"  - {cidade}: {qtd:,} registros")

# -------------------------------------------------------------------
# 4. Filtrar dados
# -------------------------------------------------------------------
# Filtrar por combustível
df_comb = df[df['Produto'] == combustivel].copy()
print(f"\nRegistros para {combustivel}: {len(df_comb):,}")

# Filtrar pelas duas cidades
df_filtrado = df_comb[(df_comb['Municipio'] == cidade_piaui) | 
                       (df_comb['Municipio'] == cidade_outra)].copy()
print(f"Registros para as cidades {cidade_piaui} e {cidade_outra}: {len(df_filtrado):,}")

if len(df_filtrado) == 0:
    print(f"\nERRO: Nenhum dado encontrado!")
    print("Verifique se o produto e as cidades existem na base.")
    print("Use as listas acima para escolher valores válidos.")
    exit()

# -------------------------------------------------------------------
# 5. Preparar dados para análise
# -------------------------------------------------------------------
# Converter preço (substituir vírgula por ponto)
df_filtrado['Valor de Venda'] = df_filtrado['Valor de Venda'].astype(str).str.replace(',', '.').astype(float)

# Converter data
df_filtrado['Data da Coleta'] = pd.to_datetime(df_filtrado['Data da Coleta'], dayfirst=True)

# Criar coluna de semana (início da semana)
df_filtrado['semana'] = df_filtrado['Data da Coleta'].dt.to_period('W').apply(lambda r: r.start_time)

print(f"\nPeríodo dos dados: {df_filtrado['Data da Coleta'].min().date()} a {df_filtrado['Data da Coleta'].max().date()}")

# -------------------------------------------------------------------
# 6. Construir base pareada por semana
# -------------------------------------------------------------------
# Média por semana para cada cidade
df_semanal = df_filtrado.groupby(['semana', 'Municipio'])['Valor de Venda'].mean().reset_index()

# Pivotar para ter as duas cidades como colunas
df_pivot = df_semanal.pivot(index='semana', columns='Municipio', values='Valor de Venda')

# Remover semanas onde falta dados de alguma cidade
df_pivot = df_pivot.dropna()

print(f"\nSemanas com dados completos para ambas as cidades: {len(df_pivot)}")

if len(df_pivot) < 30:
    print(f"\nATENÇÃO: Apenas {len(df_pivot)} semanas disponíveis (mínimo recomendado: 30)")
    print("Os resultados podem não ser conclusivos.")
    
    if len(df_pivot) < 2:
        print("Dados insuficientes para análise. Tente outras cidades ou combustível.")
        exit()

# Extrair as séries de preços
preco_a = df_pivot[cidade_piaui]
preco_b = df_pivot[cidade_outra]

# Criar DataFrame pareado
df_pareado = pd.DataFrame({
    'semana': df_pivot.index,
    f'preco_{cidade_piaui}': preco_a.values,
    f'preco_{cidade_outra}': preco_b.values,
    'diferenca': preco_a.values - preco_b.values
})
df_pareado = df_pareado.sort_values('semana').reset_index(drop=True)

# -------------------------------------------------------------------
# 7. Estatísticas descritivas
# -------------------------------------------------------------------
def estatisticas_descritivas(serie, nome):
    stats = {
        'Média (R$)': serie.mean(),
        'Mediana (R$)': serie.median(),
        'Mínimo (R$)': serie.min(),
        'Máximo (R$)': serie.max(),
        'Q1 - 1º Quartil (R$)': serie.quantile(0.25),
        'Q3 - 3º Quartil (R$)': serie.quantile(0.75),
        'Desvio Padrão (R$)': serie.std(),
        'Coef. Variação (%)': (serie.std() / serie.mean()) * 100
    }
    return pd.Series(stats, name=nome)

desc_a = estatisticas_descritivas(preco_a, cidade_piaui)
desc_b = estatisticas_descritivas(preco_b, cidade_outra)

tabela_descritiva = pd.concat([desc_a, desc_b], axis=1)

print("\n" + "="*80)
print("TABELA DE MEDIDAS DESCRITIVAS")
print("="*80)
print(tabela_descritiva.to_string())

# -------------------------------------------------------------------
# 8. Teste t pareado
# -------------------------------------------------------------------
alpha = 0.05
t_stat, p_valor = ttest_rel(preco_a, preco_b)

print("\n" + "="*80)
print("TESTE T PAREADO")
print("="*80)
print(f"H0 (Hipótese nula): A média das diferenças semanais é igual a zero.")
print(f"H1 (Hipótese alternativa): A média das diferenças semanais é diferente de zero.")
print(f"\nNível de significância (α) = {alpha}")
print(f"Estatística t calculada = {t_stat:.4f}")
print(f"Valor-p = {p_valor:.6f}")
print(f"Valor crítico (bicaudal) = ±{np.abs(np.percentile(np.random.standard_t(len(preco_a)-1, 10000), [2.5, 97.5]))[1]:.4f}")

print("\n" + "-"*40)
if p_valor < alpha:
    decisao = "REJEITAR H0"
    conclusao = f"Existe diferença estatisticamente significativa entre os preços de {combustivel} nas duas cidades."
    print(f"✓ DECISÃO: {decisao}")
else:
    decisao = "NÃO REJEITAR H0"
    conclusao = f"Não há evidência estatística suficiente para afirmar diferença entre os preços de {combustivel} nas duas cidades."
    print(f"✗ DECISÃO: {decisao}")
print("-"*40)

# -------------------------------------------------------------------
# 9. Gráficos
# -------------------------------------------------------------------
# Histograma - Cidade A
fig, ax = plt.subplots()
ax.hist(preco_a, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(preco_a.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Média: R$ {preco_a.mean():.2f}')
ax.axvline(preco_a.median(), color='green', linestyle='dashed', linewidth=2, label=f'Mediana: R$ {preco_a.median():.2f}')
ax.set_xlabel('Preço (R$)')
ax.set_ylabel('Frequência')
ax.set_title(f'Histograma - Preço de {combustivel}\n{cidade_piaui} (PI)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('histograma_cidade_piaui.png', dpi=150, bbox_inches='tight')
plt.show()

# Histograma - Cidade B
fig, ax = plt.subplots()
ax.hist(preco_b, bins=15, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(preco_b.mean(), color='red', linestyle='dashed', linewidth=2, label=f'Média: R$ {preco_b.mean():.2f}')
ax.axvline(preco_b.median(), color='green', linestyle='dashed', linewidth=2, label=f'Mediana: R$ {preco_b.median():.2f}')
ax.set_xlabel('Preço (R$)')
ax.set_ylabel('Frequência')
ax.set_title(f'Histograma - Preço de {combustivel}\n{cidade_outra}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('histograma_cidade_outra.png', dpi=150, bbox_inches='tight')
plt.show()

# Boxplot comparativo
dados_box = pd.melt(df_pareado[[f'preco_{cidade_piaui}', f'preco_{cidade_outra}']], 
                     var_name='Cidade', value_name='Preço (R$)')
dados_box['Cidade'] = dados_box['Cidade'].str.replace('preco_', '')

fig, ax = plt.subplots()
sns.boxplot(x='Cidade', y='Preço (R$)', data=dados_box, palette='Set2', ax=ax)
ax.set_title(f'Comparação de Preços - {combustivel}')
ax.set_ylabel('Preço (R$)')
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('boxplot_comparativo.png', dpi=150, bbox_inches='tight')
plt.show()

# Gráfico de séries temporais
fig, ax = plt.subplots()
ax.plot(df_pareado['semana'], preco_a, marker='o', linewidth=2, markersize=4, 
        label=cidade_piaui, color='steelblue')
ax.plot(df_pareado['semana'], preco_b, marker='s', linewidth=2, markersize=4, 
        label=cidade_outra, color='coral')
ax.set_xlabel('Semana')
ax.set_ylabel('Preço Médio (R$)')
ax.set_title(f'Evolução Semanal dos Preços - {combustivel}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('serie_temporal.png', dpi=150, bbox_inches='tight')
plt.show()

# -------------------------------------------------------------------
# 10. Salvar resultados
# -------------------------------------------------------------------
# Salvar base pareada
df_pareado.to_csv('base_pareada_semanas.csv', index=False, sep=';', decimal=',')
print(f"\n✓ Base pareada salva: 'base_pareada_semanas.csv'")

# Salvar estatísticas
tabela_descritiva.to_csv('estatisticas_descritivas.csv', sep=';', decimal=',')

# -------------------------------------------------------------------
# 11. Relatório final para o artigo
# -------------------------------------------------------------------
print("\n" + "="*80)
print("RELATÓRIO FINAL - PARA O ARTIGO")
print("="*80)
print(f"""
ANÁLISE DE PREÇOS DE {combustivel.upper()}

Cidades analisadas:
  • {cidade_piaui} (Piauí)
  • {cidade_outra}

Período analisado: {df_pareado['semana'].min().date()} a {df_pareado['semana'].max().date()}
Número de semanas com dados pareados: {len(df_pareado)}

=== MEDIDAS DESCRITIVAS ===
{cidade_piaui}:
  • Preço médio: R$ {preco_a.mean():.2f}
  • Mediana: R$ {preco_a.median():.2f}
  • Desvio padrão: R$ {preco_a.std():.2f}
  • Coeficiente de variação: {(preco_a.std()/preco_a.mean()*100):.1f}%

{cidade_outra}:
  • Preço médio: R$ {preco_b.mean():.2f}
  • Mediana: R$ {preco_b.median():.2f}
  • Desvio padrão: R$ {preco_b.std():.2f}
  • Coeficiente de variação: {(preco_b.std()/preco_b.mean()*100):.1f}%

Diferença média entre as cidades: R$ {(preco_a.mean() - preco_b.mean()):.2f}

=== TESTE DE HIPÓTESE ===
Hipótese nula (H0): Não há diferença significativa entre os preços.
Hipótese alternativa (H1): Há diferença significativa entre os preços.

Resultados do teste t pareado:
  • Estatística t: {t_stat:.4f}
  • Valor-p: {p_valor:.6f}
  • Nível de significância: α = 0.05

Decisão estatística: {decisao}
Conclusão: {conclusao}

=== GRÁFICOS GERADOS ===
1. histograma_cidade_piaui.png
2. histograma_cidade_outra.png
3. boxplot_comparativo.png
4. serie_temporal.png

=== ARQUIVOS GERADOS ===
1. base_pareada_semanas.csv - Dados semanais pareados
2. estatisticas_descritivas.csv - Tabela de estatísticas
""")

print("="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO!")
print("="*80)