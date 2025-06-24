// Aguarda o DOM carregar completamente
document.addEventListener("DOMContentLoaded", function () {
    const dados = window.dadosEstatisticas;

    const selectCampeonato = document.getElementById("filtroCampeonato");
    const selectTime = document.getElementById("filtroTime");

    selectCampeonato.addEventListener("change", () => {
        const campeonato = selectCampeonato.value;
        const filtrado = campeonato ? dados.filter(j => j["Campeonato"] === campeonato) : dados;
        renderEstatisticasCampeonato(filtrado);
    });

    selectTime.addEventListener("change", () => {
        const time = selectTime.value;
        const filtrado = time ? dados.filter(j => j["Time Referência"] === time) : [];
        renderEstatisticasTime(filtrado, time);
    });

    // Inicializa com todos os dados do campeonato
    renderEstatisticasCampeonato(dados);
});

// Estatísticas gerais do campeonato (bloco 1)
function renderEstatisticasCampeonato(jogos) {
    const bloco = document.getElementById("bloco-campeonato");
    bloco.innerHTML = "";

    const totalJogos = jogos.length;
    const totalGols = soma(jogos.map(j => j["Num Gols"]));
    const mediaGols = (totalGols / totalJogos).toFixed(2);

    const totalAmarelos = soma(jogos.map(j => j["Cartões Amarelos"]));
    const totalVermelhos = soma(jogos.map(j => j["Cartões Vermelhos"]));

    // Construir painel estatístico
    bloco.innerHTML = `
        <div class="card p-4 bg-light mb-4">
            <h4>Resumo do Campeonato</h4>
            <ul>
                <li><b>Total de Jogos:</b> ${totalJogos}</li>
                <li><b>Total de Gols:</b> ${totalGols}</li>
                <li><b>Média de Gols por Jogo:</b> ${mediaGols}</li>
                <li><b>Total de Cartões Amarelos:</b> ${totalAmarelos}</li>
                <li><b>Total de Cartões Vermelhos:</b> ${totalVermelhos}</li>
            </ul>
        </div>
        <div id="grafico-campeonato-gols" style="height: 400px;"></div>
    `;

    // Agrupamento por time
    const stats = {};
    jogos.forEach(j => {
        const time = j["Time Referência"];
        if (!stats[time]) stats[time] = { pro: 0, contra: 0 };
        stats[time].pro += parseFloat(j["Gols Casa"]);
        stats[time].contra += parseFloat(j["Gols Visitante"]);
    });

    const times = Object.keys(stats);
    const golsPro = times.map(t => stats[t].pro);
    const golsContra = times.map(t => stats[t].contra);

    const layout = {
        title: "Gols Marcados vs Sofridos",
        barmode: 'group',
        margin: { t: 40 },
        paper_bgcolor: '#f8f9fa',
        plot_bgcolor: '#f8f9fa',
        font: { color: '#000' }
    };

    const traces = [
        { x: golsPro, y: times, name: 'Gols Marcados', orientation: 'h', type: 'bar' },
        { x: golsContra, y: times, name: 'Gols Sofridos', orientation: 'h', type: 'bar' }
    ];

    Plotly.newPlot("grafico-campeonato-gols", traces, layout);
}

// Estatísticas por time (bloco 2)
function renderEstatisticasTime(jogos, time) {
    const bloco = document.getElementById("bloco-time");
    bloco.innerHTML = "";

    if (!jogos.length) {
        bloco.innerHTML = `<div class="alert alert-warning">Nenhum jogo encontrado para o time selecionado.</div>`;
        return;
    }

    const totalJogos = jogos.length;
    const totalGols = soma(jogos.map(j => j["Num Gols"]));
    const mediaGols = (totalGols / totalJogos).toFixed(2);

    const totalAmarelos = soma(jogos.map(j => j["Cartões Amarelos"]));
    const totalVermelhos = soma(jogos.map(j => j["Cartões Vermelhos"]));

    bloco.innerHTML = `
        <div class="card p-4 bg-light mb-4">
            <h4>Resumo do ${time}</h4>
            <ul>
                <li><b>Total de Jogos:</b> ${totalJogos}</li>
                <li><b>Total de Gols:</b> ${totalGols}</li>
                <li><b>Média de Gols por Jogo:</b> ${mediaGols}</li>
                <li><b>Total de Cartões Amarelos:</b> ${totalAmarelos}</li>
                <li><b>Total de Cartões Vermelhos:</b> ${totalVermelhos}</li>
            </ul>
        </div>
        <div id="grafico-time-gols" style="height: 400px;"></div>
    `;

    // Linha temporal: gols por jogo
    const datas = jogos.map(j => j["Data e Hora"]);
    const gols = jogos.map(j => j["Num Gols"]);

    const trace = {
        x: datas,
        y: gols,
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Gols',
        line: { shape: 'spline', color: '#007bff' }
    };

    const layout = {
        title: `Evolução dos Gols por Jogo - ${time}`,
        paper_bgcolor: '#f8f9fa',
        plot_bgcolor: '#f8f9fa',
        font: { color: '#000' },
        margin: { t: 40 }
    };

    Plotly.newPlot("grafico-time-gols", [trace], layout);
}

// Função auxiliar
function soma(lista) {
    return lista.reduce((acc, val) => acc + (parseFloat(val) || 0), 0);
}

// BOTÃO DE ATUALIZAÇÃO DOS DADOS
const btn = document.getElementById("btnAtualizarDados");
const barra = document.getElementById("barraProgresso");
const info = document.getElementById("infoAtualizacao");
const progressoBarra = document.getElementById("progressoInterno");

btn.addEventListener("click", function () {
    fetch("/atualizar-dados", { method: "POST" });

    barra.style.display = "block";
    info.style.display = "block";
    progressoBarra.style.width = "0%";
    progressoBarra.innerText = "0%";

    const interval = setInterval(() => {
        fetch("/progresso")
            .then(response => response.json())
            .then(data => {
                const valor = data.progresso;
                progressoBarra.style.width = valor + "%";
                progressoBarra.innerText = valor + "%";

                if (valor >= 100) {
                    clearInterval(interval);
                    info.innerText = "Atualização concluída!";
                    setTimeout(() => location.reload(), 2000);
                }
            });
    }, 500);
});