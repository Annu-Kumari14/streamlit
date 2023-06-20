import streamlit as st
from PIL import Image
import pandas as pd
from io import StringIO
import time
import base64
from xgboost import XGBClassifier
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from feature_user import vae_cvae_synthetic_generation, generate_synthetic_data,vae_generated_synthetic_data,generate_synthetic_data_vae,copulagan,fast_ml,gaussian_copula,ctgan,tvae,convert_df
import streamlit.components.v1 as components
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from sdv.single_table import CopulaGANSynthesizer
from sdv.lite import SingleTablePreset
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.single_table import CTGANSynthesizer
from sdv.single_table import TVAESynthesizer
from sdmetrics.reports.single_table import QualityReport
from streamlit import download_button
import json
from functools import reduce



# Define the pages
pages = {
    "About the App":"Providing an info to the user about the app",
    "Upload-generate-check-quality-score": "Upload-generate-check-quality-score"
    # "Validation": "Validate Synthetic Data",
    # "Visualization": "Visualize your data"
}


# Define the current page
current_page = st.sidebar.radio("Navigation", list(pages.keys()))

if current_page == "About the App":
    st.title("Hey user!")
    st.header("A walkthrough to follow the steps:")
    # col1, col2, col3 = st.columns(3)
    tab1, tab2, tab3 = st.tabs(["Step 1", "Step 2", "Step 3"])

    with tab1:

        st.subheader("Upload your real data")
        st.image("data1.png")
        expander = st.expander("Click Me for more info")
        expander.write("Hi User! This app will help you to generate synthetic data with different synthesizers. Just be ready with your real data!")

    with tab2:
        st.subheader("Generate the synthetic data")
        st.image("gd.png")
        expander=st.expander("Click Me")
        expander.write("When you click the button Generate and save synthetic data,it will basically generate all type of synthetic data based on different synthesizers and save the data on your local machine.")
        expander.write("Conditional Variational Autoencoders (CVAE) is use to generate synthetic data samples. Refer the link below to read about it more.")
        expander.markdown("https://towardsdatascience.com/understanding-conditional-variational-autoencoders-cd62b4f57bf8")
        expander.write("A Variational Autoencoder (VAE) is a deep learning model that can generate new data samples.")
        expander.markdown("https://www.scaler.com/topics/deep-learning/variational-autoencoder/")
        expander.image("vae.png")
        expander.write("Now let's introduce some synthesizers which we have used for generation of synthetic data:")

        expander.write("The Copula GAN Synthesizer uses a mix classic, statistical methods and GAN-based deep learning methods to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/copulagansynthesizer")
        expander.write("The Fast ML Preset synthesizer is optimized for modeling speed.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/fast-ml-preset")
        expander.write("The Gaussian Copula Synthesizer uses classic, statistical methods to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/gaussiancopulasynthesizer")
        expander.write("The CTGAN Synthesizer uses GAN-based, deep learning methods to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/ctgansynthesizer")
        expander.write("The TVAE Synthesizer uses a variational autoencoder (VAE)-based, neural network techniques to train a model and generate synthetic data.")
        expander.markdown("https://docs.sdv.dev/sdv/single-table-data/modeling/synthesizers/tvaesynthesizer")

    with tab3:
        st.subheader("Check Quality Report")
        st.image("score.jpg")
        expander=st.expander("Click Me")
        expander.write("Quality report evaluates the shapes of the columns (marginal distributions) and the pairwise trends between the columns (correlations). Refer the below link to read about it more")
        expander.markdown("https://docs.sdv.dev/sdmetrics/reports/quality-report/whats-included")
    # expander = st.expander("Click Me for more info")
    # expander.write(
    #     "Hi User! This app will help you to generate synthetic data with different synthesizers. Just be ready with your real data!"
        
    # )
    


# Render the content based on the current page
elif current_page == "Upload-generate-check-quality-score":
    st.title("Welcome to the world of generating synthetic data!")
    image = Image.open('header.png')
    st.image(image)
    # st.image(image, caption='Synthetic Data Generation')
    uploaded_file = st.file_uploader("Please upload your real data")
    if uploaded_file is not None:
        st.success('Data uploaded Successfully!', icon="✅")
    # To read file as bytes:
        bytes_data = uploaded_file.getvalue()
    # st.write(bytes_data)

    # To convert to a string based IO:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # st.write(stringio)

    # To read file as string:
        string_data = stringio.read()
    # st.write(string_data)
   
        dataframe= pd.read_csv(uploaded_file)
        st.write(dataframe)
        st.write('Shape of real data :',dataframe.shape)
        # link_text = "Click here to know more about the input columns"
        # toggle_code = """
        # <script>
        # function toggleExpander() {
        #     var expander = document.getElementById("expander");
        #     expander.classList.toggle("hidden");
        # }
        # </script>
        # """
        # st.markdown(f'<a href="javascript:toggleExpander()">{link_text}</a>', unsafe_allow_html=True)
        # with st.expander("Expanded Information"):
        #     st.write("This is the expanded information.")
        # container = st.container()
        # container.markdown(f'<a href="javascript:void(0)" onclick="if(document.getElementById(\'expander\').style.display === \'none\') {{ document.getElementById(\'expander\').style.display = \'block\'; }} else {{ document.getElementById(\'expander\').style.display = \'none\'; }}">{link_text}</a>', unsafe_allow_html=True)
        # expander_visible = container.columns(1)[0].checkbox("Click me to know more about input columns", key="expander")
        # if expander_visible:
        #     st.write("Input column.")



        select_input_column = st.multiselect('Select input columns for CVAE',options=list(dataframe.columns),key="1")
        st.write(select_input_column)
        # st.sidebar.title("Preview")
        # for option in select_input_column:
        #     st.sidebar.write(option)
        container = st.container()
        expander_visible = container.columns(1)[0].checkbox("Click the checkbox to know more about conditional columns", key="5")
        if expander_visible:
            st.write("A conditional column in CVAE is a column of data that is used to condition the output of the model. For example if a VIN/user  has only 3 preferences of Genre in real data and you want the same 3 prefernces in your synthetic data as well then VIN and Genre will come in your conditional input.")
            # st.write("Make sure you don't input VIN/USER_ID/NAME")
            st.write("Columns entered in input column can't be re-entered.")

        select_conditional_column = st.multiselect('Select conditional columns for CVAE',options=list(dataframe.columns))
        st.write(select_conditional_column)
        container = st.container()
        expander_visible1 = container.columns(1)[0].checkbox("Click the checkbox to know more about input columns for VAE", key="6")
        if expander_visible1:
            st.write("Input columns can be features/labels/targets/ouputs.")
          

        select_input_column_vae = st.multiselect('Select input columns for VAE',options=list(dataframe.columns),key="2")
        st.write(select_input_column_vae)
        # st.info('Primary key should be the unique identified column such as user_id/VIN/Name. Metadata is a set of data that describes the data itself. This includes information such as the data types, the column names, and the relationships between the columns. Metadata is used by the synthesizer to generate synthetic data that is similar to the real data.', icon="ℹ️")
        # pk_metadata=st.selectbox('Select your primary key to generate metadata',options=list(dataframe.columns))
        # st.write(pk_metadata)
        lr_rate=st.number_input('Input the learning rate',min_value=0.01)
        st.write(lr_rate)
        container = st.container()
        expander_visible2 = container.columns(1)[0].checkbox("Click the checkbox to know more about Latent Dimension", key="7")
        if expander_visible2:
            st.write("The latent dimension is defined by the number of dimensions of the mean and variance vectors. Each dimension in the latent space can be considered as a latent variable or latent feature that captures certain characteristics of the input data. By choosing an appropriate latent dimension, the VAE can learn a compact representation of the data that captures the most important aspects while discarding unnecessary details.")
            st.write("The choice of latent dimension is a hyperparameter that needs to be tuned during the training process to achieve a balance between representation power and simplicity.")
            st.write("NOTE- preferably the latent space should be less than that of the input dimension")
        latent_dims=st.number_input('Input the latent dimension',min_value=20)
        st.write(latent_dims)
        select_epoch=st.number_input('Input the epoch',min_value=1)
        st.write(select_epoch)
        select_batchsize=st.number_input('Input the batch size',min_value=1)
        st.write(select_batchsize)
        samples = st.slider('How many number of rows you want to generate?',0,250,500)
        st.write(samples)
        # st.info("Generating Synthetic Data...")


        # synthetic_cvae=pd.DataFrame()
        # synthetic_vae=pd.DataFrame()
        # synthetic_copula=pd.DataFrame()
        # synthetic_fastml=pd.DataFrame()
        # synthetic_gaussian=pd.DataFrame()
        # synthetic_ctgan=pd.DataFrame()
        # synthetic_tvae=pd.DataFrame()

        synthetic_cvae=None
        synthetic_vae=None
        synthetic_copula=None
        synthetic_fastml=None
        synthetic_gaussian=None
        synthetic_ctgan=None
        synthetic_tvae=None

        # syn1=pd.DataFrame()
        # syn2=pd.DataFrame()
        # syn3=pd.DataFrame()
        # syn4=pd.DataFrame()
        # syn5=pd.DataFrame()
        # syn6=pd.DataFrame()
        # syn7=pd.DataFrame()


        
        # def generate_synthetic_datas():   
        #     global dataframe 
        #     global synthetic_cvae
        #     global synthetic_vae
        #     global synthetic_copula
        #     global synthetic_fastml
        #     global synthetic_gaussian
        #     global synthetic_ctgan
        #     global synthetic_tvae
        #     uploader_file = st.file_uploader("Upload Data") 
        #     if uploader_file is not None: 
        #         dataframe = pd.read_csv(uploader_file)
        if st.button('Generate Synthetic Data'):
            start_time = time.time()
            with st.spinner('Loading...'):
                time.sleep(78)
            # st.write('For CVAE')
            one_hot_encoder,condition_encoder,dataframe,features,condition_features,select_input_column,select_conditional_column,encoder,decoder,latent_dims,condition_data,encoded_features = vae_cvae_synthetic_generation(dataframe,select_input_column,select_conditional_column,lr_rate,latent_dims,select_epoch,select_batchsize)
            synthetic_cvae = generate_synthetic_data(one_hot_encoder,condition_encoder,samples,dataframe,features,condition_features,select_input_column,select_conditional_column,encoder,decoder,latent_dims,condition_data,encoded_features)
            # synthetic_cvae=pd.concat([synthetic_cvae,syn1])
        
            # st.write(synthetic)
            st.write('Shape of synthetic data cvae:',synthetic_cvae.shape)
            # csv = convert_df(synthetic)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv,
            #     file_name='sd_cvae.csv',
            #     mime='text/csv',
            #     )
            # synthetic.to_csv('synthetic_data_cvae.csv', index=False)
            st.success('CVAE Synthetic Data generated successfully.')

            # st.stop()

            # st.write('For VAE')
            # select_input_column_vae = st.multiselect('Select input columns for VAE',options=list(dataframe.columns),key="2")
            # st.write(select_input_column_vae)
            enc,encoded_cols,df,select_input_column_vae,encoder,decoder,latent_dim=vae_generated_synthetic_data(dataframe,select_input_column_vae,lr_rate,latent_dims,select_epoch,select_batchsize)
            synthetic_vae=generate_synthetic_data_vae(enc,encoded_cols,samples,dataframe,select_input_column_vae,encoder,decoder,latent_dims)
            # synthetic_vae=pd.concat([synthetic_vae,syn2])
            # st.write(synthetic_vae)
            st.write('Shape of synthetic data vae:',synthetic_vae.shape)
            # csv1 = convert_df(synthetic_vae)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv1,
            #     file_name='sd_vae.csv',
            #     mime='text/csv',
            #     )
            # synthetic_vae.to_csv('synthetic_data_vae.csv', index=False)
            st.success(' VAE Synthetic Data generated successfully.')

            # st.write('For CopulaGAN')
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=dataframe)
            metadata.validate()
            synthesizer = CopulaGANSynthesizer(metadata)
            synthetic_copula=copulagan(dataframe,synthesizer,samples)
            # synthetic_copula=pd.concat([synthetic_copula,syn3])
            # st.write(synthetic_copula)
            st.write('Shape of synthetic data copulagan:',synthetic_copula.shape)
            # csv2 = convert_df(synthetic_copula)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv2,
            #     file_name='sd_copula.csv',
            #     mime='text/csv',
            #     )
            # synthetic_copula.to_csv('synthetic_data_copula.csv', index=False)
            st.success('CopulaGAN Synthetic Data generated successfully.')

            # st.write('For FAST_ML')
            synthesizer_fastml = SingleTablePreset(metadata, name='FAST_ML')
            synthetic_fastml=fast_ml(dataframe,synthesizer_fastml,samples)
            # synthetic_fastml=pd.concat([synthetic_fastml,syn4])
            # st.write(synthetic_fastml)
            st.write('Shape of synthetic data fast_ml:',synthetic_fastml.shape)
            # csv3 = convert_df(synthetic_fastml)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv3,
            #     file_name='sd_fastml.csv',
            #     mime='text/csv',
            #     )
            # synthetic_fastml.to_csv('synthetic_data_fastml.csv', index=False)
            st.success(' Fast_ML Synthetic Data generated successfully.')

            # st.write('GaussianCopula')
            synthesizer_gaussian = GaussianCopulaSynthesizer(metadata)
            synthetic_gaussian=gaussian_copula(dataframe,synthesizer_gaussian,samples)
            # synthetic_gaussian=pd.concat([synthetic_gaussian,syn5])
            # st.write(synthetic_gaussian)
            st.write('Shape of synthetic data gaussian copula:',synthetic_gaussian.shape)
            # csv4 = convert_df(synthetic_gaussian)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv4,
            #     file_name='sd_gaussian.csv',
            #     mime='text/csv',
            #     )
            # synthetic_gaussian.to_csv('synthetic_data_gaussian_copula.csv', index=False)
            st.success(' Gaussian Copula Synthetic Data generated successfully.')

            # st.write('CTGAN')
            synthesizer_ctgan = CTGANSynthesizer(metadata)
            synthetic_ctgan=ctgan(dataframe,synthesizer_ctgan,samples)
            # synthetic_ctgan=pd.concat([synthetic_ctgan,syn6])
            # st.write(synthetic_ctgan)
            st.write('Shape of synthetic data CTGAN:',synthetic_ctgan.shape)
            # csv5 = convert_df(synthetic_ctgan)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv5,
            #     file_name='sd_ctgan.csv',
            #     mime='text/csv',
            #     )
            # synthetic_ctgan.to_csv('synthetic_data_CTGAN.csv', index=False)
            st.success('CTGAN Synthetic Data generated successfully.')

            # st.write('TVAE')
            synthesizer_tvae = TVAESynthesizer(metadata)
            synthetic_tvae=tvae(dataframe,synthesizer_tvae,samples)
            # synthetic_tvae=pd.concat([synthetic_tvae,syn7])
            # st.write(synthetic_tvae)
            st.write('Shape of synthetic data TVAE:',synthetic_tvae.shape)
            # csv6 = convert_df(synthetic_tvae)
            # st.download_button(
            #     label="Download CSV",
            #     data=csv6,
            #     file_name='sd_tvae.csv',
            #     mime='text/csv',
            #     )
            # synthetic_tvae.to_csv('synthetic_data_TVAE.csv', index=False)
            st.success('TVAE Synthetic Data generated successfully.')



            # st.button('Check Quality Score')
                # start_time = time.time()
            # dataframe=dataframe.drop('VIN',axis=1)
                # synthetic=pd.read_csv('synthetic_data_cvae.csv')
            # synthetic0=synthetic_cvae.drop('VIN',axis=1)

                # synthetic1=pd.read_csv('synthetic_data_vae.csv')
            # synthetic1=synthetic_vae.drop('VIN',axis=1)

                # synthetic2=pd.read_csv('synthetic_data_copula.csv')
            # synthetic2=synthetic_copula.drop('VIN',axis=1)

                # synthetic3=pd.read_csv('synthetic_data_fastml.csv')
            # synthetic3=synthetic_fastml.drop('VIN',axis=1)

                # synthetic4=pd.read_csv('synthetic_data_gaussian_copula.csv')
            # synthetic4=synthetic_gaussian.drop('VIN',axis=1)

                # synthetic5=pd.read_csv('synthetic_data_CTGAN.csv')
            # synthetic5=synthetic_ctgan.drop('VIN',axis=1)

                # synthetic6=pd.read_csv('synthetic_data_TVAE.csv')
            # synthetic6=synthetic_tvae.drop('VIN',axis=1)

            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(data=dataframe)
            metadata.validate()
            quality_report = evaluate_quality(
                dataframe,
                synthetic_cvae,
                metadata)
            a=(f"{round((quality_report.get_score()) * 100)}%")
            x1=quality_report.get_details(property_name='Column Shapes')

            quality_report = evaluate_quality(
                dataframe,
                synthetic_vae,
                metadata
                )
            b=(f"{round((quality_report.get_score()) * 100)}%")
            x2=quality_report.get_details(property_name='Column Shapes')

            quality_report = evaluate_quality(
                dataframe,
                synthetic_copula,
                metadata
                )
            c=(f"{round((quality_report.get_score()) * 100)}%")
            x3=quality_report.get_details(property_name='Column Shapes')

            quality_report = evaluate_quality(
                dataframe,
                synthetic_fastml,
                metadata
                )
            d=(f"{round((quality_report.get_score()) * 100)}%")
            x4=quality_report.get_details(property_name='Column Shapes')

            quality_report = evaluate_quality(
                dataframe,
                synthetic_gaussian,
                metadata
                )
            e=(f"{round((quality_report.get_score()) * 100)}%")
            x5=quality_report.get_details(property_name='Column Shapes')

            quality_report = evaluate_quality(
                dataframe,
                synthetic_ctgan,
                metadata
                )
            f=(f"{round((quality_report.get_score()) * 100)}%")
            x6=quality_report.get_details(property_name='Column Shapes')

            quality_report = evaluate_quality(
                dataframe,
                synthetic_tvae,
                metadata
                )
            g=(f"{round((quality_report.get_score()) * 100)}%")
            x7=quality_report.get_details(property_name='Column Shapes')


                
            scores_df = pd.DataFrame({ 'Model': ['CVAE','VAE','CopulaGAN', 'FAST_ML', 'Gaussian Copula', 'CTGAN', 'TVAE'], 'Quality Score': [a, b, c, d, e, f, g] })  

            scores_df_sorted = scores_df.sort_values('Quality Score',ascending= False)
            st.dataframe(scores_df_sorted)
                # st.line_chart(scores_df_sorted, x='Model', y='Quality Score')
                # scores_df_sorted['Quality Score'] = scores_df_sorted['Quality Score'].sort_index(ascending=True)
                # scores_df_sorted['Quality Score'] = scores_df_sorted['Quality Score'][::-1] 
            st.line_chart(scores_df_sorted, x='Model', y='Quality Score')

        
            model_names = ['CVAE','VAE','CopulaGAN', 'FAST_ML', 'Gaussian Copula', 'CTGAN', 'TVAE']
            reshaped_df = pd.DataFrame(columns=['Column', 'Metric'])
            dfs = [x1, x2, x3, x4, x5, x6, x7]
            for df, model in zip(dfs, model_names):
                df['Quality Score of ' + model] = df['Quality Score']
                reshaped_df = pd.concat([reshaped_df, df['Quality Score of ' + model]], axis=1)
            reshaped_df['Column'] = dfs[0]['Column']
            reshaped_df['Metric'] = dfs[0]['Metric']
            st.write(reshaped_df)
            var=reshaped_df['Column']
            reshaped_df=reshaped_df.set_index('Column')
            reshaped_df=reshaped_df.drop(columns='Metric', axis=1)
            st.line_chart(reshaped_df)


            check_winner = {'CVAE': a,'VAE': b ,'CopulaGAN': c, 'Fast_ML': d,'Gaussian Copula':e,'CTGAN':f,'TVAE':g}
            winner = max(check_winner, key=check_winner.get)
            st.markdown("<h4 style='text-align: left; color: #e4b016;'>The best quality score is: {}</h4>".format(winner), unsafe_allow_html=True)
            end_time = time.time()
            execution_time = end_time - start_time
            st.write(f"Execution time: {execution_time:.2f} seconds")
            if winner == 'CVAE':

                csv = convert_df(synthetic_cvae)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_cvae.csv',
                    mime='text/csv',
                    )
            elif winner == 'VAE':

                csv = convert_df(synthetic_vae)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_vae.csv',
                    mime='text/csv',
                    )

            elif winner == 'CopulaGAN':

                csv = convert_df(synthetic_copula)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_copula.csv',
                    mime='text/csv',
                    )
                
            elif winner == 'FAST_ML':
                csv = convert_df(synthetic_fastml)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_fastml.csv',
                    mime='text/csv',
                    )
                    
            elif winner == 'Gaussian Copula':
                csv = convert_df(synthetic_gaussian)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_gaussian.csv',
                    mime='text/csv',
                    )
                    
            elif winner == 'CTGAN':
                csv = convert_df(synthetic_ctgan)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_ctgan.csv',
                    mime='text/csv',
                    )
                    
            elif winner == 'TVAE':
                csv = convert_df(synthetic_tvae)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name='synthetic_data_tvae.csv',
                    mime='text/csv',
                    )
            

            # download_options = {
            #     'CVAE':synthetic_cvae,
            #     'VAE':synthetic_vae,
            #     'CopulaGAN':synthetic_copula,
            #     'FAST_ML':synthetic_fastml,
            #     'Gaussian Copula':synthetic_gaussian,
            #     'CTGAN':synthetic_ctgan,
            #     'TVAE':synthetic_tvae
            #     }
            # if 'selected_option_column' not in st.session_state:
            #     st.session_state.selected_option_column = None   
            # download_option = st.selectbox("Which synthetic data do you want to download?", list(download_options.keys()))
            # selected_dataframe = download_options[download_option]
            # st.session_state.selected_option_column = selected_dataframe
            # stored_option_column = st.session_state.selected_option_column
            
            # st.write(stored_option_column)
            # csv = convert_df(stored_option_column)
            # st.download_button(
            #         label="Download CSV",
            #         data=csv,
            #         file_name='synthetic_data.csv',
            #         mime='text/csv',
            #         )

            









            

                # return synthetic_cvae,synthetic_vae,synthetic_copula,synthetic_fastml,synthetic_gaussian,synthetic_ctgan,synthetic_tvae

        # synthetic_cvae,synthetic_vae,synthetic_copula,synthetic_fastml,synthetic_gaussian,synthetic_ctgan,synthetic_tvae = generate_synthetic_datas()
        # cvae_sd=synthetic_cvae
        # vae_sd=synthetic_vae
        # copula_sd=synthetic_copula
        # fast_sd=synthetic_fastml
        # gaussian_sd=synthetic_gaussian
        # ctgan_sd=synthetic_ctgan
        # tvae_sd=synthetic_tvae
            
            # download_options = {
            #     'CVAE':synthetic_cvae,
            #     'VAE':synthetic_vae,
            #     'CopulaGAN':synthetic_copula,
            #     'FAST_ML':synthetic_fastml,
            #     'Gaussian Copula':synthetic_gaussian,
            #     'CTGAN':synthetic_ctgan,
            #     'TVAE':synthetic_tvae
            #     }

        # download_options = {
        #     'CVAE':cvae_sd,
        #     'VAE':vae_sd,
        #     'CopulaGAN':copula_sd,
        #     'FAST_ML':fast_sd,
        #     'Gaussian Copula':gaussian_sd,
        #     'CTGAN':ctgan_sd,
        #     'TVAE':tvae_sd
        #     }

           
        # st.write("syn3:",syn3)
        # st.write("syn4:",syn4)
        # st.write("syn5:",syn5)
        # st.write("syn6:",syn6)
        # st.write("syn7:",syn7)

        # st.write(synthetic_cvae)
        # st.write(synthetic_vae)
        # st.write(synthetic_copula)
        # st.write(synthetic_fastml)
        # st.write(synthetic_gaussian)
        # st.write(synthetic_ctgan)
        # st.write(synthetic_tvae)

        # st.write('synthetic_copula:',synthetic_copula)
        # st.write('synthetic_fastml:',synthetic_fastml)
        # st.write('synthetic_gaussian:',synthetic_gaussian)
        # st.write('synthetic_ctgan:',synthetic_ctgan)
        # st.write('synthetic_tvae:',synthetic_tvae)


        # download_option = st.selectbox("Which synthetic data do you want to download?", list(download_options.keys()))
        # if st.button("Download CSV"):
        #     selected_dataframe = download_options[download_option]
        #     st.write(selected_dataframe)
        #     if selected_dataframe is not None:
        #         csv = selected_dataframe.to_csv(index=False)
        #         b64 = base64.b64encode(csv.encode()).decode()
        #         href = f'<a href="data:file/csv;base64,{b64}" download="{download_option}.csv">Download {download_option}</a>'
        #         st.markdown(href, unsafe_allow_html=True)
        #     else:
        #         st.write("Selected synthetic dataset is not available.")

            # download_option = st.selectbox("Which synthetic data do you want to download?", list(download_options.keys()))
            # selected_dataframe = download_options[download_option]
            # st.write(selected_dataframe)
            # csv = convert_df(selected_dataframe)
            # st.download_button(
            #         label="Download CSV",
            #         data=csv,
            #         file_name='synthetic_data.csv',
            #         mime='text/csv',
            #         )




        # download_option=st.selectbox("Which synthetic data you want to download?",
        # ('CVAE', 'VAE', 'CopulaGAN', 'FAST_ML', 'Gaussian Copula', 'CTGAN', 'TVAE'))
        # st.write(download_option)

        # if download_option == 'CVAE':
        #     csv = convert_df(synthetic)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_cvae.csv',
        #         mime='text/csv',
        #         )
        # elif download_option == 'VAE':
        #     csv = convert_df(synthetic_vae)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_vae.csv',
        #         mime='text/csv',
        #         )
                
        # elif download_option == 'CopulaGAN':
        #     csv = convert_df(synthetic_copula)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_copula.csv',
        #         mime='text/csv',
        #         )
                
        # elif download_option == 'FAST_ML':
        #     csv = convert_df(synthetic_fastml)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_fastml.csv',
        #          mime='text/csv',
        #         )
                
        # elif download_option == 'Gaussian Copula':
        #     csv = convert_df(synthetic_gaussian)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_gaussian.csv',
        #         mime='text/csv',
        #         )
                
        # elif download_option == 'CTGAN':
        #     csv = convert_df(synthetic_ctgan)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_ctgan.csv',
        #         mime='text/csv',
        #         )
                
        # elif download_option == 'TVAE':
        #     csv = convert_df(synthetic_tvae)
        #     st.download_button(
        #         label="Download CSV",
        #         data=csv,
        #         file_name='synthetic_data_tvae.csv',
        #         mime='text/csv',
        #         )
                

            
                            
                




