# Python In-built packages
from pathlib import Path
import PIL

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Deteccion de Plagas en la agricultura Mexicana",
    # page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
# st.title("Deteccion de Plagas en la agricultura Mexicana")
# st.caption('Updload a photo with this :blue[hand signals]: :+1:, :hand:, :i_love_you_hand_sign:, and :spock-hand:.')
# st.caption('Then click the :blue[Detect Objects] button and check the result.')

# barra Desplazadora
st.sidebar.header("Configracion del Modelo")

# Model Options
model_type = st.sidebar.radio(
    "Seleccionar Tarea", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Seleccione la Confianza del Modelo", 25, 100, 40)) / 100

# Seleccionar Deteccion 
if model_type == 'Deteccion':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Cargar el modelo de aprendizaje entrenado
# model_path
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Selecione fuente", settings.SOURCES_LIST)

source_img = None
# Seleccion de Imagen
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Seleccione Imagen...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Se produjo un error al abrir la carpeta")
            st.error(ex)

    with col2:        
            if st.sidebar.button('Detectar objetos'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Resultados de la detección"):
                        for box in boxes:
                # Se puede obtener informacion de data para dezplegar info de la plaga 
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("Aún no se ha subido ninguna imagen")
                # quitar el video y youtube para despuer realizar 
# elif source_radio == settings.VIDEO:
#    helper.play_stored_video(confidence, model)

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

# elif source_radio == settings.YOUTUBE:
#    helper.play_youtube_video(confidence, model)

else:
    st.error("Seleccione un tipo de fuente válido")
