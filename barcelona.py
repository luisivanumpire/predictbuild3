import streamlit as st # type: ignore
import pandas as pd # type: ignore
import joblib
import numpy as np


# Título de la aplicación
st.title("PREDICTBUILD")
st.header("Predicción de Viviendas con IA")

# Instrucciones para el usuario
st.write("Ingrese las características de las viviendas para predecir el precio.")

# Front 
m2 = float(st.text_input("Metros Cuadrados", value="100"))
num_habitaciones = int(st.slider('Número de habitaciones :',0 ,10 ,1))
## Tipo
df_tipo = pd.read_csv('tipo_vivienda.csv')
opciones_tipo = df_tipo['tipo_vivienda'].tolist()
seleccion_tipo = st.selectbox('Tipo de vivienda:', opciones_tipo)
indice_tipo = opciones_tipo.index(seleccion_tipo)
valor_tipo_vivienda = df_tipo.loc[indice_tipo, 'name_val']
## Mostrar la opción seleccionada
#st.write(f'Has seleccionado tipo: {seleccion_tipo}')
#st.write(f'Índice del tipo : {indice_tipo}')
#st.write(f'Valor del tipo de v.: {valor_tipo_vivienda}')

## sub area
df = pd.read_csv('subarea_list.csv')
#st.write(f'Has seleccionado: {df.head(1)}')
opciones = df['subarea'].tolist()
seleccion_subarea = st.selectbox('Seleccione un barrio:', opciones)
indice_seleccionado = opciones.index(seleccion_subarea)
valor_m2_subarea = df.loc[indice_seleccionado, 'valm2']

## Mostrar la opción seleccionada
#st.write(f'Has seleccionado: {seleccion_subarea}')
#st.write(f'Índice de la selección: {indice_seleccionado}')
#st.write(f'Valor por metro 2: {valor_m2_subarea}')

# Creamos el array de entrada
X_list =    [m2, 
              num_habitaciones,
              float(valor_tipo_vivienda),
              int(valor_m2_subarea)
              ]

#X = np.array([float(elemento) for elemento in X_list])
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)

# Botón para ejecutar el modelo
if st.button("Predecir"):
    if len(X) > 0:
        
        # Cargar el modelo y los parámetros de normalización guardados
        #scaler = joblib.load('scaler.pkl')
        model = joblib.load('modelo_random_forest_joblib.pkl')
        
        # Mostrar las primeras filas del DataFrame cargado
        #st.write("Datos cargados:")
        #st.write(X)
        
        #data_scaled = scaler.transform(X)
        
        # Realizar predicciones con el modelo XGBoost
        predicciones = model.predict(X)
        
        # Mostrar las predicciones
        st.write("Predicciones del presupuesto estimado:")
        st.write(predicciones)

