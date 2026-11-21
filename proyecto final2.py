# ==============================================================================
# PROYECTO: Simulador de Atencion Hospitalaria / Clinica
# CURSO: Analisis de Algoritmos y Estrategias de Programacion (UPN 2026-2)
# ENTORNO: Python 3 + Visual Studio Code
# PARAMETROS DEL ESCENARIO:
#   - 10 doctores disponibles en un lapso de 5 horas laborales (3,000 min-medico)
#   - 200 pacientes registrados en el turno
#   - Minimo 15 minutos por consulta/identificacion del paciente
#   - Registro detallado por tipo de prioridad de triaje
# ESTRATEGIAS:
#   1. Divide y Venceras (Merge Sort para clasificacion y triaje de pacientes)
#   2. Programacion Dinamica (Asignacion optima de recursos finitos - Mochila 0/1)
# ==============================================================================

import random
import time
import tracemalloc
from dataclasses import dataclass
from typing import Dict, List, Tuple


# Catalogo oficial de tipos de prioridad registrados en triaje
TIPOS_PRIORIDAD: Dict[int, str] = {
    5: "Prioridad I - Emergencia Critica",
    4: "Prioridad II - Urgencia Alta",
    3: "Prioridad III - Urgencia Media",
    2: "Prioridad IV - Consulta Prioritaria",
    1: "Prioridad V - Consulta General",
}


# ------------------------------------------------------------------------------
# MODELO DE DATOS (Pacientes y Recursos de la Clinica/Hospital)
# ------------------------------------------------------------------------------
@dataclass
class Paciente:
    id_paciente: int
    nombre: str
    prioridad_triaje: int   # 1 a 5 segun TIPOS_PRIORIDAD
    tipo_prioridad: str     # Descripcion registrada del nivel de triaje
    tiempo_atencion: int    # Minimo 15 min para correcta identificacion y consulta
    requiere_cama: bool     # True si requiere cama/observacion
    requiere_equipo: bool   # True si requiere equipo de diagnostico
    llegada_min: int        # Minuto de ingreso dentro del turno (0 a 300 min)


@dataclass
class RecursosClinica:
    medicos_disponibles: int
    horas_laborales: int
    minutos_turno_total: int   # medicos * horas * 60
    camas_disponibles: int
    equipos_disponibles: int


# ==============================================================================
# CONCEPTO 1: DIVIDE Y VENCERAS
# Aplicacion: Organizacion y ordenamiento de pacientes por prioridad de triaje
# (de mayor a menor urgencia) y tiempo de llegada (de menor a mayor).
# Divide la lista de pacientes en mitades independientes, ordena cada mitad
# recursivamente y combina (merge) los resultados ordenados.
# Complejidad: O(n log n) en tiempo | O(n) en espacio auxiliar.
# ==============================================================================
def divide_y_venceras_ordenar(pacientes: List[Paciente]) -> List[Paciente]:
    if len(pacientes) <= 1:
        return pacientes

    medio = len(pacientes) // 2
    izquierda = divide_y_venceras_ordenar(pacientes[:medio])
    derecha = divide_y_venceras_ordenar(pacientes[medio:])

    return _combinar_pacientes(izquierda, derecha)


def _combinar_pacientes(izq: List[Paciente], der: List[Paciente]) -> List[Paciente]:
    resultado = []
    i = j = 0

    while i < len(izq) and j < len(der):
        # Prioridad mas alta primero; en empate, el que llego antes
        if (izq[i].prioridad_triaje > der[j].prioridad_triaje) or (
            izq[i].prioridad_triaje == der[j].prioridad_triaje
            and izq[i].llegada_min <= der[j].llegada_min
        ):
            resultado.append(izq[i])
            i += 1
        else:
            resultado.append(der[j])
            j += 1

    resultado.extend(izq[i:])
    resultado.extend(der[j:])
    return resultado


def atender_con_divide_y_venceras(
    pacientes: List[Paciente], recursos: RecursosClinica
) -> Tuple[List[Paciente], int, int]:
    """
    Ordena por Divide y Venceras y asigna recursos secuencialmente
    respetando la disponibilidad de minutos-medico, camas y equipos.
    """
    ordenados = divide_y_venceras_ordenar(pacientes)

    atendidos = []
    tiempo_restante = recursos.minutos_turno_total
    camas_restantes = recursos.camas_disponibles
    equipos_restantes = recursos.equipos_disponibles
    tiempo_espera_acumulado = 0
    # Reloj paralelo aproximado segun la cantidad de doctores disponibles
    carga_medicos = [0] * recursos.medicos_disponibles

    for p in ordenados:
        puede_cama = (not p.requiere_cama) or (camas_restantes > 0)
        puede_equipo = (not p.requiere_equipo) or (equipos_restantes > 0)

        if p.tiempo_atencion <= tiempo_restante and puede_cama and puede_equipo:
            # Asignar al doctor que se desocupa mas pronto
            idx_medico = carga_medicos.index(min(carga_medicos))
            inicio_atencion = max(p.llegada_min, carga_medicos[idx_medico])

            # Verificar que la consulta termine dentro de las 5 horas laborales (300 min)
            limite_turno_min = recursos.horas_laborales * 60
            if inicio_atencion + p.tiempo_atencion <= limite_turno_min:
                atendidos.append(p)
                tiempo_restante -= p.tiempo_atencion
                if p.requiere_cama:
                    camas_restantes -= 1
                if p.requiere_equipo:
                    equipos_restantes -= 1

                tiempo_espera_acumulado += inicio_atencion - p.llegada_min
                carga_medicos[idx_medico] = inicio_atencion + p.tiempo_atencion

    puntaje_prioridad = sum(p.prioridad_triaje for p in atendidos)
    return atendidos, puntaje_prioridad, tiempo_espera_acumulado


# ==============================================================================
# CONCEPTO 2: PROGRAMACION DINAMICA
# Aplicacion: Optimizacion global de asignacion de recursos (capacidad de
# tiempo medico disponible) para maximizar la prioridad clinica atendida.
# Almacena subproblemas resueltos en una tabla dp[i][t] y reconstruye
# exactamente que pacientes deben ser atendidos (backtracking).
# Complejidad: O(n * C) donde C = minutos_turno_total.
# ==============================================================================
def programacion_dinamica_asignar(
    pacientes: List[Paciente], recursos: RecursosClinica
) -> Tuple[List[Paciente], int, int]:
    """
    Resuelve la seleccion optima de pacientes dentro de la capacidad de
    tiempo medico (minutos_turno_total), maximizando la prioridad total.
    Luego asigna doctores y verifica camas/equipos disponibles.
    """
    n = len(pacientes)
    capacidad_t = recursos.minutos_turno_total

    # Tabla DP: dp[i][t] = maxima prioridad usando los primeros i pacientes con tiempo t
    dp = [[0] * (capacidad_t + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        p = pacientes[i - 1]
        duracion = p.tiempo_atencion
        valor = p.prioridad_triaje

        for t in range(capacidad_t + 1):
            dp[i][t] = dp[i - 1][t]
            if duracion <= t:
                dp[i][t] = max(dp[i][t], dp[i - 1][t - duracion] + valor)

    # Reconstruccion de la solucion optima (Backtracking sobre la tabla DP)
    seleccionados = []
    t = capacidad_t
    for i in range(n, 0, -1):
        if dp[i][t] != dp[i - 1][t]:
            p = pacientes[i - 1]
            seleccionados.append(p)
            t -= p.tiempo_atencion

    seleccionados.reverse()
    # Ordenar los seleccionados por llegada para simular la atencion en el turno
    seleccionados.sort(key=lambda x: (x.llegada_min, -x.prioridad_triaje))

    atendidos = []
    camas_restantes = recursos.camas_disponibles
    equipos_restantes = recursos.equipos_disponibles
    tiempo_espera_acumulado = 0
    carga_medicos = [0] * recursos.medicos_disponibles
    limite_turno_min = recursos.horas_laborales * 60

    for p in seleccionados:
        puede_cama = (not p.requiere_cama) or (camas_restantes > 0)
        puede_equipo = (not p.requiere_equipo) or (equipos_restantes > 0)
        if puede_cama and puede_equipo:
            idx_medico = carga_medicos.index(min(carga_medicos))
            inicio_atencion = max(p.llegada_min, carga_medicos[idx_medico])

            if inicio_atencion + p.tiempo_atencion <= limite_turno_min:
                atendidos.append(p)
                if p.requiere_cama:
                    camas_restantes -= 1
                if p.requiere_equipo:
                    equipos_restantes -= 1

                tiempo_espera_acumulado += inicio_atencion - p.llegada_min
                carga_medicos[idx_medico] = inicio_atencion + p.tiempo_atencion

    puntaje_prioridad = sum(p.prioridad_triaje for p in atendidos)
    return atendidos, puntaje_prioridad, tiempo_espera_acumulado


# ==============================================================================
# GENERADOR DE ESCENARIOS SIMULADOS Y MEDICION DE RENDIMIENTO
# ==============================================================================
def generar_pacientes(cantidad: int = 200, horas_turno: int = 5, semilla: int = 42) -> List[Paciente]:
    """
    Genera 200 pacientes en un turno de 5 horas (300 minutos).
    Cada paciente requiere un minimo de 15 minutos de consulta para su
    correcta identificacion y atencion medica.
    """
    random.seed(semilla)
    lista = []
    min_max_llegada = (horas_turno * 60) - 15  # Hasta 15 min antes del cierre

    for i in range(1, cantidad + 1):
        nivel = random.choices([5, 4, 3, 2, 1], weights=[10, 20, 30, 25, 15])[0]
        lista.append(
            Paciente(
                id_paciente=i,
                nombre=f"Paciente_{i:03d}",
                prioridad_triaje=nivel,
                tipo_prioridad=TIPOS_PRIORIDAD[nivel],
                # Minimo 15 minutos obligatorios para identificacion y consulta
                tiempo_atencion=random.choice([15, 15, 20, 25, 30, 45]),
                requiere_cama=random.random() < 0.25,
                requiere_equipo=random.random() < 0.30,
                llegada_min=random.randint(0, min_max_llegada),
            )
        )
    return lista


def medir_estrategia(nombre: str, funcion, pacientes: List[Paciente], recursos: RecursosClinica):
    tracemalloc.start()
    inicio = time.perf_counter()

    atendidos, puntaje, espera_total = funcion(pacientes, recursos)

    fin = time.perf_counter()
    _, pico_memoria = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tiempo_ms = (fin - inicio) * 1000
    memoria_kb = pico_memoria / 1024
    espera_prom = espera_total / len(atendidos) if atendidos else 0.0
    uso_tiempo = sum(p.tiempo_atencion for p in atendidos)
    aprovechamiento = (uso_tiempo / recursos.minutos_turno_total) * 100

    # Conteo por tipo de prioridad registrado
    conteo_prioridad: Dict[int, int] = {nivel: 0 for nivel in sorted(TIPOS_PRIORIDAD.keys(), reverse=True)}
    for p in atendidos:
        conteo_prioridad[p.prioridad_triaje] += 1

    print(f"\n--- {nombre} ---")
    print(f"Pacientes atendidos : {len(atendidos)} / {len(pacientes)}")
    print(f"Prioridad total     : {puntaje} pts")
    print("Registro por tipo de prioridad atendida:")
    for nivel, cantidad in conteo_prioridad.items():
        print(f"  - {TIPOS_PRIORIDAD[nivel]}: {cantidad} pacientes")
    print(f"Tiempo de espera    : {espera_prom:.2f} min (promedio)")
    print(f"Uso de tiempo medico: {uso_tiempo}/{recursos.minutos_turno_total} min ({aprovechamiento:.1f}%)")
    print(f"Tiempo de ejecucion : {tiempo_ms:.4f} ms")
    print(f"Pico de memoria     : {memoria_kb:.2f} KB")


# ==============================================================================
# EJECUCION PRINCIPAL
# ==============================================================================
# ==============================================================================
# EJECUCION PRINCIPAL DIVIDIDA EN 2 PARTES
# ==============================================================================
def configurar_escenario() -> Tuple[List[Paciente], RecursosClinica]:
    """Inicializa los recursos de la clinica y los 200 pacientes del turno."""
    doctores = 10
    horas_laborales = 5
    recursos = RecursosClinica(
        medicos_disponibles=doctores,
        horas_laborales=horas_laborales,
        minutos_turno_total=doctores * horas_laborales * 60,  # 3,000 min-medico
        camas_disponibles=60,
        equipos_disponibles=60,
    )
    pacientes = generar_pacientes(cantidad=200, horas_turno=horas_laborales)
    return pacientes, recursos


# ------------------------------------------------------------------------------
# PARTE 1: DIVIDE Y VENCERAS
# ------------------------------------------------------------------------------
def ejecutar_parte_1_divide_y_venceras(
    pacientes: List[Paciente], recursos: RecursosClinica
) -> None:
    print("====================================================================")
    print("   PARTE 1: EJECUCION CON DIVIDE Y VENCERAS (MERGE SORT TRIAJE)")
    print("   Turno: 5 horas | 10 Doctores | 200 Pacientes | Min: 15 min")
    print("====================================================================")
    medir_estrategia(
        "Estrategia 1: Divide y Venceras (Merge Sort Triaje)",
        atender_con_divide_y_venceras,
        pacientes,
        recursos,
    )


# ------------------------------------------------------------------------------
# PARTE 2: PROGRAMACION DINAMICA
# ------------------------------------------------------------------------------
def ejecutar_parte_2_programacion_dinamica(
    pacientes: List[Paciente], recursos: RecursosClinica
) -> None:
    print("\n====================================================================")
    print("   PARTE 2: EJECUCION CON PROGRAMACION DINAMICA (MOCHILA 0/1)")
    print("   Turno: 5 horas | 10 Doctores | 200 Pacientes | Min: 15 min")
    print("====================================================================")
    medir_estrategia(
        "Estrategia 2: Programacion Dinamica (Optimizacion DP)",
        programacion_dinamica_asignar,
        pacientes,
        recursos,
    )


if __name__ == "__main__":
    escenario_pacientes, recursos_clinica = configurar_escenario()

    # 1. Ejecutar Parte 1 (Divide y Venceras)
    ejecutar_parte_1_divide_y_venceras(escenario_pacientes, recursos_clinica)

    # 2. Ejecutar Parte 2 (Programacion Dinamica)
    ejecutar_parte_2_programacion_dinamica(escenario_pacientes, recursos_clinica)