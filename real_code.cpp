#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>

#define NUM_INPUT 4
#define NUM_NEURAL 4 
#define NUM_OUTPUT 1

// Moi mang deu bat dau tu so 1 nen kich co cong them 1
double data[1000],cost[1000],loss[1000];
double lr = 0.2;
int order[1000];
int n,d;
double x[NUM_INPUT+1],b[NUM_NEURAL+1],a[NUM_NEURAL+1],z[NUM_NEURAL+1];
double ao,bo,zo,y,min,max;		// bo = b cua ouput, ao = a cua output, zo = z cua ouput 
double w[NUM_NEURAL+1][NUM_NEURAL+2];
int nn = NUM_NEURAL,ni = NUM_INPUT,no = NUM_OUTPUT;
double ss_z[NUM_NEURAL+1],ss_a[NUM_NEURAL+1],ss_w[NUM_NEURAL+1][NUM_NEURAL+2],ss_b[NUM_NEURAL+1],ss_bo,ss_ao,ss_zo;
char inputfile[50],weightfile[50] = "weight.out";

// ham dung de in ra ket qua
void out_result(double ans){
	printf("\n  **************RESULT***************\n");
	printf("\n  ___________________________________\n");
	printf("  | 	  DU LIEU DU DOAN LA   	    |\n");
	printf("  |_________________________________|\n");
	printf("  |   %2d   |      %12.lf      |\n",n+1,ans);
	printf("  |________|________________________|\n");
}

// ham dung de in ra bang du lieu nhap vao
void out_data(){		 
	printf("\n  ___________________________________\n");
	printf("  | 	BANG DU LIEU DA NHAP   	    |\n");
	printf("  |_________________________________|\n");
	printf("  |   STT  |       DOANH THU        |\n");
	printf("  |________|________________________|\n");
	for (int i=1;i<=n;i++){
		printf("  |   %2d   |      %12.lf      |\n",order[i],data[i]);
		printf("  |________|________________________|\n");
	}
}

double sigmoid(double t){
	return 1/(1+exp(-t));
}

// dung de lay input tu file
void input_file(double data[], int order[], int &n, char tenFile[50]){		
	FILE *f;
	f = fopen(tenFile,"r");
	if(f==NULL){
		printf("\n Loi Mo File \n");
		return;
	}int i=1;
	while(!feof(f)){
	// lay input tu file
		fscanf(f,"%d",&order[i]);
		fscanf(f,"%lf",&data[i]);
		i++;	
	}fclose(f);
	n = i-2;
}

// dung de in ra w b bo ra file
void out_file(char tenFile[100]){
	FILE *f;
	f = fopen(tenFile, "a");
	if(f==NULL){
		printf("\n Loi mo file \n");
		return;
	}
	
	int i,j;
	for(i=1;i<=nn;i++){			// in ra file tu w[1][1] toi w[1][4] sau do la b[1] ......b[2]
		for(j=1;j<=nn;j++){
			fprintf(f,"%lf   ",w[i][j]);
		}fprintf(f,"%lf   ",b[i]);		
	}for(i=1;i<=nn;i++){
		fprintf(f, "%lf   ",w[i][nn+1]);	// in ra file weight tu hidden layer toi outputlayer
	}fprintf(f,"%lf   \n",bo);
	fclose(f);
}

// dung de random weight va cac bias
void rand_weight(double w[][NUM_NEURAL+2],double b[],double &bo, int nn){
	int i,j;
	srand((int)time(0));
	double r ;
	for(i=1;i<=nn;i++){
		for(j=1;j<=nn+1;j++){
			r = 10 + rand()%(99 - 10);
			r /= 100;
			w[i][j] = r;
		}
	}for(i=1;i<=nn;i++){
		r = 10 + rand()%(99 - 10);
		r /= 100;
		b[i] = r;
	}r = 10 + rand()%(99-10);
	r /= 100;
	bo = r;
}

// xu ly so lieu dua ve 0 va 1
void proccess_data(double data[],int n){				
	double max = data[1];
	int i;
	d = 0;
	for(i=1;i<=n;i++){		// tim so lon nhat trong data
		if(data[i]>=max){
			max = data[i];
		}
	}
	while(max>=1){			// tinh so chu so cua max
		max /= 10;
		d++;
	}
	for(i=1;i<=n;i++){		// quy doi tat ca ve chay tu so 0 toi 1
		data[i] /= pow(10,d);
	}
}

// lay ra 4 input tu mang data
void take_input(int k,double x[],double data[],int ni){		
	int i,j=1;
	for(i=k;i<k+ni;i++){
		x[j]=data[i];
		j++;
	}y = data[k+4];
}

//progation process
void propagation(){
		// tinh z va a
	double sum,pt ;
	int i,j;
	for(i=1;i<=nn;i++){
		sum = 0;
		for(j=1;j<=nn;j++){
			pt = w[j][i]*x[j];
			sum += pt;
		}z[i] = sum + b[i];
		a[i] = sigmoid(z[i]);
	}
	
		// tinh zo va ao
	sum = 0;
	for(i=1;i<=nn;i++){
		pt = w[i][nn+1]*a[i];
		sum += pt;
	}zo = sum + bo;
	ao = sigmoid(zo);
}

// update weight process (backpropagation)
void update_weight(){
	int i,j;
	
	// ouput layer -> hidden layer
	ss_zo = ao - y;
	for(i=1;i<=nn;i++){
		w[i][nn+1] = w[i][nn+1] - lr*a[i]*ss_zo;		//ss_w[i][nn+1] = a[i]*ss_zo
		ss_a[i] = w[i][nn+1]*ss_zo;
	}ss_bo = ss_zo;			
	
	//hidden layer -> input layer
	for(i=1;i<=nn;i++){
		ss_z[i] = a[i]*(1-a[i])*ss_a[i];
		ss_b[i] = ss_z[i];
	}for(i=1;i<=nn;i++){
		for(j=1;j<=nn;j++){
			ss_w[i][j] = x[i]*ss_z[j];
		}
	}
	
	// update weight
	for(i=1;i<=nn;i++){
		for(j=1;j<=nn+1;j++){
			w[i][j] -= lr*ss_w[i][j];
		}
	}
	// update bias
	for(i=1;i<=nn;i++){
		b[i] -= lr*ss_b[i];
	}bo -= lr*ss_bo;
}

// training process
void train(char tenFile[50]){
	int i,j;
	rand_weight(w,b,bo,nn);
	out_file(tenFile);
	for(i=1;i<=n-ni;i++){
		take_input(i,x,data,ni);
		propagation();
		update_weight();
		out_file(tenFile);
	}
}

// thuc hien tat ca qua trinh training
void proccess_1(){
		lr = lr * 10;
		//lay du lieu
		printf("\n  Moi ban nhap ten file chua du lieu: ");
		gets(inputfile);
		input_file(data,order,n,inputfile);
		
		out_data();
			
			// xu ly so lieu 
		proccess_data(data,n);
		
			//training	
		train(weightfile);
}

// lay ra du lieu cua 4 thang cuoi de du doan thang tiep theo
void take_input_2(){		
	int i;
	for(i=1;i<=ni;i++){
		x[i] = data[n-ni+i];
	}
}

//lay weight va bias duoc training nhu lan truoc do
void take_weight(char tenFile[50]){		
	FILE *f;
	f = fopen(tenFile, "r");
	if(f==NULL){
		printf("\n Loi mo file \n");
		return;
	}
	
	int i,j;
	for(i=1;i<=nn;i++){		
		for(j=1;j<=nn;j++){
			fscanf(f,"%lf",&w[i][j]);
		}fscanf(f,"%lf",&b[i]);
	}for(i=1;i<=nn;i++){
		fscanf(f,"%lf",&w[i][nn+1]);
	}fscanf(f,"%lf",&bo);
	fclose(f);
}

// dung de du doan so lieu cua thang cuoi cung
void proccess_2(){
	printf("\n==========================================");
	printf("\n*   Du doan so lieu tu du lieu da tinh   *");
	printf("\n==========================================\n");
		// lay du lieu
	take_input_2();
		
		// lay weight
	take_weight(weightfile);
		
		// tinh toan va dua ra du doan
	propagation();
	
	double ans = ao*pow(10,d);
	out_result(ans);
}

int main(){
	printf("==============================================================\n");
	printf("	     	  DO AN LAP TRINH PBL1  					  \n");
	printf("==============================================================\n");
	printf("Ma de: 503\n");	
	printf("De tai: BAI TOAN DU DOAN DOANH SO BAN HANG SU DUNG MANG NEURAL\n");
	printf("Giao vien huong dan:  PGS.TS.NGUYEN TAN KHOI\n");
	printf("Ho va ten sinh vien:  NGUYEN VAN DUNG	       MSSV: 102210356\n");
	printf("		      NGUYEN HOANG NHAT MINH   MSSV: 102210361\n");
	printf("Lop hoc phan: 21TCLC_NHAT2		       Nhom: 21Nh99\n\n");
	
	printf("**************************************************************\n");
	printf("	CHUONG TRINH DU DOAN DOANH THU QUA MANG NEURAL 	  \n");
	printf("**************************************************************\n");
	
	char check='S';
	while(check != 'D'){
		proccess_1();
		printf("\n  Du lieu ban nhap dung chua (D/S): ");
		scanf("%c",&check);
		printf("\n--------------------------------\n");
		fflush(stdin);
	}
	
	proccess_2();
	
	return 0;
}
