/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkMeshFileWriter_h
#define itkMeshFileWriter_h

#include "itkMeshFileWriterException.h"
#include "itkProcessObject.h"
#include "itkMeshIOBase.h"

namespace itk
{

/**
 * \class MeshFileWriter
 * \brief Writes mesh data to a single file.
 *
 * MeshFileWriter writes its input data to a single output file.
 * MeshFileWriter interfaces with an MeshIO class to write out the
 * data.
 *
 * A plugable factory pattern is used that allows different kinds of writers
 * to be registered (even at run time) without having to modify the
 * code in this class. You can either manually instantiate the MeshIO
 * object and associate it with the MeshFileWriter, or let the class
 * figure it out from the extension. Normally just setting the filename
 * with a suitable suffix (".vtk", etc) and setting the input
 * to the writer is enough to get the writer to work properly.
 *
 * \author Wanlin Zhu. University of New South Wales, Australia.
 *
 * \sa MeshIOBase
 *
 * \ingroup IOFilters
 * \ingroup ITKIOMeshBase
 */
template <typename TInputMesh>
class ITK_TEMPLATE_EXPORT MeshFileWriter : public ProcessObject
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(MeshFileWriter);

  /** Standard class type aliases. */
  using Self = MeshFileWriter;
  using Superclass = ProcessObject;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** \see LightObject::GetNameOfClass() */
  itkOverrideGetNameOfClassMacro(MeshFileWriter);

  /** Some convenient type alias. */
  using InputMeshType = TInputMesh;
  using InputMeshPointer = typename InputMeshType::Pointer;
  using InputMeshRegionType = typename InputMeshType::RegionType;
  using InputMeshPixelType = typename InputMeshType::PixelType;
  using InputMeshCellType = typename InputMeshType::CellType;
  using SizeValueType = typename MeshIOBase::SizeValueType;

  /** Set/Get the mesh input of this writer.  */
  using Superclass::SetInput;
  void
  SetInput(const InputMeshType * input);

  const InputMeshType *
  GetInput();

  const InputMeshType *
  GetInput(unsigned int idx);

  /** Specify the name of the output file to write. */
  /** @ITKStartGrouping */
  itkSetStringMacro(FileName);
  itkGetStringMacro(FileName);
  /** @ITKEndGrouping */
  /** Set/Get the MeshIO helper class. Usually this is created via the object
   * factory mechanism that determines whether a particular MeshIO can
   * write a certain file. This method provides a way to get the MeshIO
   * instance that is created, or one can be manually set where the
   * IO factory mechanism may not work (some mesh files with non-standard filename suffix's.
   * If the user specifies the MeshIO, we assume she makes the
   * correct choice and will allow a file to be created regardless of
   * the file extension. If the factory has set the MeshIO, the
   * extension must be supported by the specified MeshIO. */
  /** @ITKStartGrouping */
  void
  SetMeshIO(MeshIOBase * io)
  {
    if (this->m_MeshIO != io)
    {
      this->Modified();
      this->m_MeshIO = io;
    }
    m_FactorySpecifiedMeshIO = false;
    m_UserSpecifiedMeshIO = true;
  }
  itkGetModifiableObjectMacro(MeshIO, MeshIOBase);
  /** @ITKEndGrouping */
  void
  SetFileTypeAsASCII()
  {
    m_FileTypeIsBINARY = false;
  }
  void
  SetFileTypeAsBINARY()
  {
    m_FileTypeIsBINARY = true;
  }

  /** A special version of the Update() method for writers.  It
   * invokes start and end events and handles releasing data. It
   * eventually calls GenerateData() which does the actual writing. */
  virtual void
  Write();

  /** Aliased to the Write() method to be consistent with the rest of the
   * pipeline. */
  void
  Update() override
  {
    this->Write();
  }

  /** Set the compression On or Off */
  /** @ITKStartGrouping */
  itkSetMacro(UseCompression, bool);
  itkGetConstReferenceMacro(UseCompression, bool);
  itkBooleanMacro(UseCompression);
  /** @ITKEndGrouping */
protected:
  MeshFileWriter() = default;
  ~MeshFileWriter() override = default;
  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  template <typename Output>
  void
  CopyPointsToBuffer(Output * data);

  template <typename Output>
  void
  CopyCellsToBuffer(Output * data);

  template <typename Output>
  void
  CopyPointDataToBuffer(Output * data);

  template <typename Output>
  void
  CopyCellDataToBuffer(Output * data);

  void
  WritePoints();

  void
  WriteCells();

  void
  WritePointData();

  void
  WriteCellData();

private:
  std::string         m_FileName{};
  MeshIOBase::Pointer m_MeshIO{ nullptr };
  bool                m_UserSpecifiedMeshIO{ false };    // track whether the MeshIO is user specified
  bool                m_FactorySpecifiedMeshIO{ false }; // track whether the factory mechanism set the MeshIO
  bool                m_UseCompression{ false };
  bool                m_FileTypeIsBINARY{ false };
};


/** Convenience function for writing a mesh.
 *
 * The mesh parameter may be a either SmartPointer or a raw pointer and const or non-const.
 * */
template <typename TMeshPointer>
ITK_TEMPLATE_EXPORT void
WriteMesh(TMeshPointer && mesh, const std::string & filename, bool compress = false)
{
  using NonReferenceMeshPointer = std::remove_reference_t<TMeshPointer>;
  static_assert(std::is_pointer_v<NonReferenceMeshPointer> || mpl::IsSmartPointer<NonReferenceMeshPointer>::Value,
                "WriteMesh requires a raw pointer or SmartPointer.");

  using MeshType = std::remove_const_t<std::remove_reference_t<decltype(*mesh)>>;
  auto writer = MeshFileWriter<MeshType>::New();
  writer->SetInput(mesh);
  writer->SetFileName(filename);
  writer->SetUseCompression(compress);
  writer->Update();
}

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkMeshFileWriter.hxx"
#endif

#if defined ITK_MESHIO_FACTORY_REGISTER_MANAGER || defined ITK_IO_FACTORY_REGISTER_MANAGER
#  include "itkMeshIOFactoryRegisterManager.h"
#endif

#endif // itkMeshFileWriter_h
